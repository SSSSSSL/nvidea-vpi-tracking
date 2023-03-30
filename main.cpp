
#include <opencv2/core/version.hpp>
#if CV_MAJOR_VERSION >= 3
#   include <opencv2/imgcodecs.hpp>
#   include <opencv2/videoio.hpp>
#else
#   include <opencv2/highgui/highgui.hpp>
#endif
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>


#include <vpi/Array.h>
#include <vpi/Context.h>
#include <vpi/Image.h>
#include <vpi/Stream.h>
#include <vpi/algo/KLTBoundingBoxTracker.h>


#include <stdexcept>
#include <vector>
#include <iostream>

// ln -s /usr/include/opencv4/opencv2 /usr/include/opencv2

// g++ main.cpp -o tracking -std=c++11 `pkg-config --cflags opencv4 --libs` -lnvvpi

template <typename T>
void CHECK_STATUS(T STMT)
{
    VPIStatus status = STMT;
    if (status != VPI_SUCCESS)
        throw std::runtime_error(vpiStatusGetName(status));
}

static VPIImage ToVPIImage(const cv::Mat& frame)
{
    // VPIImage 생성을 위한 데이터일 뿐...인듯? 메타데이터 가지고 있는?
    VPIImageData img_data;
    memset(&img_data, 0, sizeof(img_data));

    switch (frame.type())
    {
    case CV_16U:
        img_data.type = VPI_IMAGE_TYPE_U16; break;
    
    case CV_8U:
        img_data.type = VPI_IMAGE_TYPE_U8; break;
    
    default:
        throw std::runtime_error("Frame Type Not Supported");
    }

    // 차원을 의미하는 것 같음. 그레이 스케일 데이터니까 1일듯
    img_data.numPlanes              = 1;
    img_data.planes[0].width        = frame.cols;
    img_data.planes[0].height       = frame.rows;
    img_data.planes[0].rowStride    = frame.step[0];
    img_data.planes[0].data         = frame.data;

    // 실제 이미지 객체
    VPIImage img;
    CHECK_STATUS(vpiImageWrapHostMem(&img_data, 0, &img));

    return img;
}

cv::Mat fetchFrame(cv::VideoCapture& v_cap, size_t& next_frame, VPIDeviceType dev_type)
{
    cv::Mat frame;
    if (!v_cap.read(frame))
    {
        return cv::Mat();
    }

    // VPI에선 gray scale의 input만 받아들인다고 함!
    // 그렇기에 채널이 3개라면, gray scale로 변환
    if (frame.channels() == 3)
        cvtColor(frame, frame, cv::COLOR_BGR2GRAY);

    if (dev_type == VPI_DEVICE_TYPE_PVA)
    {
        // PVA는 16bit unsigned만 허용한다고 함.
        cv::Mat aux;
        frame.convertTo(aux, CV_16U);
        frame = aux;
    }
    else
    {
        assert(frame.type() == CV_8U);
    }

    // frame 수 증가시킴. reference형
    next_frame++;

    return frame;
}

static void showKLTBoxes(VPIImage img, VPIArray boxes, VPIArray preds)
{
    cv::Mat out;

    {
        VPIImageData imgdata;
        CHECK_STATUS(vpiImageLock(img, VPI_LOCK_READ, &imgdata));

        int cvtype;
        switch (imgdata.type)
        {
        case VPI_IMAGE_TYPE_U8:
            cvtype = CV_8U; break;

        case VPI_IMAGE_TYPE_S8:
            cvtype = CV_8S; break;

        case VPI_IMAGE_TYPE_U16:
            cvtype = CV_16UC1; break;

        case VPI_IMAGE_TYPE_S16:
            cvtype = CV_16SC1; break;

        default:
            throw std::runtime_error("Image type not supported");
        }

        cv::Mat cvimg(imgdata.planes[0].height, imgdata.planes[0].width, cvtype, imgdata.planes[0].data,
                      imgdata.planes[0].rowStride);

        if (cvimg.type() == CV_16U)
        {
            cvimg.convertTo(out, CV_8U);
            cvimg = out;
            out   = cv::Mat();
        }

        cvtColor(cvimg, out, cv::COLOR_GRAY2BGR);

        CHECK_STATUS(vpiImageUnlock(img));
    }


    // Bounding Box 그리기
    VPIArrayData boxdata;
    CHECK_STATUS(vpiArrayLock(boxes, VPI_LOCK_READ, &boxdata));

    VPIArrayData preddata;
    CHECK_STATUS(vpiArrayLock(preds, VPI_LOCK_READ, &preddata));

    auto *pboxes = reinterpret_cast<VPIKLTTrackedBoundingBox *>(boxdata.data);
    auto *ppreds = reinterpret_cast<VPIHomographyTransform2D *>(preddata.data);

    srand(0);
    for (size_t i = 0; i < boxdata.size; ++i)
    {
        if (pboxes[i].trackingStatus == 1)
        {
            // So that the colors assigned to bounding boxes don't change
            // when some bbox isn't tracked anymore.
            rand();
            rand();
            rand();
            continue;
        }

        float x, y, w, h;
        x = pboxes[i].bbox.xform.mat3[0][2] + ppreds[i].mat3[0][2];
        y = pboxes[i].bbox.xform.mat3[1][2] + ppreds[i].mat3[1][2];
        w = pboxes[i].bbox.width * pboxes[i].bbox.xform.mat3[0][0] * ppreds[i].mat3[0][0];
        h = pboxes[i].bbox.height * pboxes[i].bbox.xform.mat3[1][1] * ppreds[i].mat3[1][1];

        rectangle(out, cv::Rect(x, y, w, h), cv::Scalar(rand() % 256, rand() % 256, rand() % 256), 2);
    }    

    CHECK_STATUS(vpiArrayUnlock(preds));
    CHECK_STATUS(vpiArrayUnlock(boxes));

    cv::imshow("tracked", out);
    cv::waitKey(1000 / 30);
}

int main()
{
    // VPIContext를 통해서 VPI 객체를 만들면,
    // 내가 만든 VPI 객체가 뭔지 상관없이, 그냥 VPI Context만 Destory하면
    // VPI 객체가 삭제됨.
    VPIContext ctx = nullptr;


    // 인풋 
    cv::VideoCapture v_cap(0, cv::CAP_V4L2);

    // ROI 설정
    cv::Mat temp;
    v_cap.read(temp);
    auto rect = cv::selectROI(temp);

    int x = rect.x;
    int y = rect.y;
    int w = rect.width;
    int h = rect.height;
    
    // VPI Context 생성
    CHECK_STATUS(vpiContextCreate(0, &ctx));


    // VPI Context 사용
    // 해당 구문 이후로 생성되는 모든 객체는 "ctx"가 소유하게 됨.
    CHECK_STATUS(vpiContextSetCurrent(ctx));


    // 인풋 박스와 예상된 변환을 저장하기 위한 VPI Array 선언
    VPIArray inputBoxList, inputPredList;

    std::vector<VPIKLTTrackedBoundingBox> bbox;
    std::vector<VPIHomographyTransform2D> pred;
    {
        // VPI KLT Tracker가 확인할 Bounding Box 선언
        VPIKLTTrackedBoundingBox roi;
        // scale
        roi.bbox.xform.mat3[0][0]   = 1;
        roi.bbox.xform.mat3[1][1]   = 1;
        // position
        roi.bbox.xform.mat3[0][2]   = x;
        roi.bbox.xform.mat3[1][2]   = y;
        //musb be 1
        roi.bbox.xform.mat3[2][2]   = 1;

        roi.bbox.width              = w;
        roi.bbox.height             = h;
        roi.trackingStatus          = 0; // valid tracking
        roi.templateStatus          = 1; // must update


        VPIHomographyTransform2D xform  = {};
        xform.mat3[0][0]                = 1;
        xform.mat3[1][1]                = 1;
        xform.mat3[2][2]                = 1;


        bbox.push_back(roi);
        pred.push_back(xform);
    }

    // VPI Array로 변환
    VPIArrayData data   = {};
    data.type           = VPI_ARRAY_TYPE_KLT_TRACKED_BOUNDING_BOX;
    data.capacity       = bbox.capacity();
    data.size           = 1;
    data.data           = &bbox[0];
    CHECK_STATUS(vpiArrayWrapHostMem(&data, 0, &inputBoxList));

    data.type           = VPI_ARRAY_TYPE_HOMOGRAPHY_TRANSFORM_2D;
    data.data           = &pred[0];
    CHECK_STATUS(vpiArrayWrapHostMem(&data, 0, &inputPredList));


    // 디바이스 타입 생성
    VPIDeviceType devType = VPI_DEVICE_TYPE_CUDA;
    // VPI_DEVICE_TYPE_CPU
    // VPI_DEVICE_TYPE_CUDA
    // VPI_DEVICE_TYPE_INVALID
    // VPI_DEVICE_TYPE_PVA

    // 스트림을 생성 (무슨 스트림인지는 코드 분석을 해봐야할듯)
    VPIStream stream;
    CHECK_STATUS(vpiStreamCreate(devType, &stream));
    

    // 다음 프레임 
    size_t nextFrame = 0;
    
    // cv::VideoCapture를 통해 가져올 데이터 -> cv::Mat cvTemplate, cvReference
    // 첫 프레임 읽어온다. Frame 정보 등을 활용하기위해 선언.
    cv::Mat cvTemplate  = fetchFrame(v_cap, nextFrame, devType);

    // 프레임 저장할 위치
    cv::Mat cvReference;


    // VPI를 통해 처리할 데이터 -> VPIImage imgTempalte, imgReference
    // 앞서 읽어온 첫 프레임을 통해 VPIImage의 Frame 정보로 활용할 데이터 저장한다.
    VPIImage imgTemplate = ToVPIImage(cvTemplate);


    // VPI 이미지 타입 변수 선언
    VPIImageType imgType;
    // VPI 이미지 형식 가져옴
    CHECK_STATUS(vpiImageGetType(imgTemplate, &imgType));


    // 가져온 첫번째 프레임의 특성을 이용해, KLT Bounding Box Tracker Payload를 만든다.
    VPIPayload klt;
    CHECK_STATUS(vpiCreateKLTBoundingBoxTracker(stream, cvTemplate.cols, cvTemplate.rows, imgType, &klt));


    // KLT 트래커가 사용할 파라미터. 지금 바로 사용하진 않기에, 선언만 해둠.
    VPIKLTBoundingBoxTrackerParams params = {};
    params.numberOfIterationsScaling      = 20;
    params.nccThresholdUpdate             = 0.8f;
    params.nccThresholdKill               = 0.6f;
    params.nccThresholdStop               = 1.0f;
    params.maxScaleChange                 = 0.2f;
    params.maxTranslationChange           = 1.5f;
    params.trackingType                   = VPI_KLT_INVERSE_COMPOSITIONAL;


    // 현재 프레임에 대한 추정된 bounding box 저장하기 위한 array
    VPIArray outputBoxList;
    CHECK_STATUS(vpiArrayCreate(128, VPI_ARRAY_TYPE_KLT_TRACKED_BOUNDING_BOX, 0, &outputBoxList));

    // 인풋 bbox를 아웃풋 bbox에 맞추기 위한 추정되는 변환(estimated transform)를 저장하기 위한 array
    VPIArray outputEstList;
    CHECK_STATUS(vpiArrayCreate(128, VPI_ARRAY_TYPE_HOMOGRAPHY_TRANSFORM_2D, 0, &outputEstList));


    // 현재 프레임을 참조하기 위한 VPI Image 객체
    VPIImage imgReference = nullptr;


    // 현재 존재하는 Bbox의 갯수
    size_t curNumBoxes = 0;

    do
    {
        size_t curFrame = nextFrame - 1;

        showKLTBoxes(imgTemplate, inputBoxList, inputPredList);

        // 새로운 프레임 가져온다
        vpiImageDestroy(imgReference);
        cvReference = fetchFrame(v_cap, nextFrame, devType);

        // 비디오 종료
        if (cvReference.data == nullptr)
        {
            // 탈출
            break;
        }

        VPIKLTTrackedBoundingBox a;
        a.trackingStatus;

        // cv::Mat을 VPI Image로 변경
        imgReference = ToVPIImage(cvReference);

        // 과거의 frame을 기반으로 현재 프레임의 bounding box 추정
        CHECK_STATUS(vpiSubmitKLTBoundingBoxTracker(klt, imgTemplate,
                                                    inputBoxList, inputPredList,
                                                    imgReference,
                                                    outputBoxList, outputEstList,
                                                    &params));

        // 처리 완료까지 대기
        CHECK_STATUS(vpiStreamSync(stream));

        // 다음 반복(iteration)의 인풋을 적절히 설정하기 위해, output array를 잠근다.
        VPIArrayData updatedBBoxData;
        CHECK_STATUS(vpiArrayLock(outputBoxList, VPI_LOCK_READ, &updatedBBoxData));

        VPIArrayData estimData;
        CHECK_STATUS(vpiArrayLock(outputEstList, VPI_LOCK_READ, &estimData));

        auto *updated_bbox  = reinterpret_cast<VPIKLTTrackedBoundingBox *>(updatedBBoxData.data);
        auto *estim         = reinterpret_cast<VPIHomographyTransform2D *>(estimData.data);

        {
            // 트래킹이 실패했다면?
            if (updated_bbox[0].trackingStatus)
            {
                std::cout << "tracking Status Fail" << std::endl;
                // 근데 이전엔 성공했다면,
                if (bbox[0].trackingStatus == 0)
                {
                    std::cout << " Dropped " << std::endl;
                    bbox[0].trackingStatus == 1;
                }
            }

            // 바운딩 박스의 template이 변경되어야하나?
            if (updated_bbox[0].templateStatus)
            {
                std::cout << " Updated " << std::endl;

                bbox[0] = updated_bbox[0];
                bbox[0].templateStatus = 1;

                pred[0] = VPIHomographyTransform2D{};
                pred[0].mat3[0][0] = 1;
                pred[0].mat3[1][1] = 1;
                pred[0].mat3[2][2] = 1;
            }
            else
            {
                bbox[0].templateStatus = 0;

                pred[0] = estim[0];
            }

            std::cout << bbox[0].bbox.width << " x " << bbox[0].bbox.height << " : " << bbox[0].bbox.xform.mat3[0][2] << " " << bbox[0].bbox.xform.mat3[1][2] << std::endl;

            int pred_x = bbox[0].bbox.xform.mat3[0][2] + pred[0].mat3[0][2];
            std::cout << "pred x : " << bbox[0].bbox.xform.mat3[0][2] << " + " << pred[0].mat3[0][2] << std::endl;

            int pred_y = bbox[0].bbox.xform.mat3[1][2] + pred[0].mat3[1][2];
            std::cout << "pred y : " << bbox[0].bbox.xform.mat3[1][2] << " + " << pred[0].mat3[1][2] << std::endl;

            int pred_w = bbox[0].bbox.width * bbox[0].bbox.xform.mat3[0][0] * pred[0].mat3[0][0];
            std::cout << "pred w : " << bbox[0].bbox.width << " * " << bbox[0].bbox.xform.mat3[0][0] << " * " << pred[0].mat3[0][0] << std::endl;

            int pred_h = bbox[0].bbox.height * bbox[0].bbox.xform.mat3[1][1] * pred[0].mat3[1][1];
            std::cout << "pred h : " << bbox[0].bbox.height << " * " << bbox[0].bbox.xform.mat3[1][1] << " * " << pred[0].mat3[1][1] << std::endl;

            std::cout << "predicted : " << pred_x << " x " << pred_y  << "  " << pred_w << " * " << pred_h << std::endl;
        }

        CHECK_STATUS(vpiArrayUnlock(outputBoxList));
        CHECK_STATUS(vpiArrayUnlock(outputEstList));

        CHECK_STATUS(vpiArrayInvalidate(inputBoxList));
        CHECK_STATUS(vpiArrayInvalidate(inputPredList));

        std::swap(imgTemplate, imgReference);
        std::swap(cvTemplate, cvReference);
    } while(true);


    vpiContextDestroy(ctx);

    return 0;
}
