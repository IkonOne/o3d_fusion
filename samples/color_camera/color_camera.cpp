#include <iostream>
#include <vector>
#include <depthai/depthai.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

#define MIN_MATCHES 50
#define MIN_HOMOGRAPHY_POINTS 10
#define HOMOGRAPY_SMOOTHING 5

struct State {
    cv::Mat camIntrinsics;
    cv::Vec3f rotation;
    cv::Vec3f rotationAccum;
    cv::Vec3f translation;
    cv::Vec3f translationAccum;
};

int main(void) {
    State state;

    dai::Pipeline pipeline;

    auto xoutRGB = pipeline.create<dai::node::XLinkOut>();
    xoutRGB->setStreamName("rgb");

    auto camRGB = pipeline.create<dai::node::ColorCamera>();
    camRGB->setPreviewSize(800, 800);
    camRGB->setBoardSocket(dai::CameraBoardSocket::RGB);
    camRGB->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1080_P);
    camRGB->setInterleaved(false);
    camRGB->setColorOrder(dai::ColorCameraProperties::ColorOrder::RGB);
    camRGB->setFps(30);
    camRGB->preview.link(xoutRGB->input);

    auto imu = pipeline.create<dai::node::IMU>();
    imu->enableIMUSensor({
        dai::IMUSensor::GYROSCOPE_CALIBRATED,
        dai::IMUSensor::LINEAR_ACCELERATION
    }, 200);
    imu->setBatchReportThreshold(1);
    imu->setMaxBatchReports(10);
    imu->out.link(xoutRGB->input);

    dai::Device device(pipeline);

    std::cout << "Connected cameras: ";
    for (const auto& cam : device.getConnectedCameras()) {
        std::cout << static_cast<int>(cam) << " ";
        std::cout << cam << " ";
    }
    std::cout << '\n';

    std::cout << "USB Speed: " << device.getUsbSpeed() << '\n';

    const auto qRGB = device.getOutputQueue("rgb", 4, false);
    auto cal = device.readCalibration();
    // state.camIntrinsics = cal.getCameraIntrinsics(dai::CameraBoardSocket::RGB);
    state.camIntrinsics = state.camIntrinsics.eye(3, 3, CV_32F);

    //-- ORB
    // -- https://docs.opencv.org/3.4/dc/d16/tutorial_akaze_tracking.html
    const cv::Scalar blue(255, 0, 0, 255);
    const cv::Scalar green(0, 255, 0, 255);
    const cv::Scalar red(0, 0, 255, 255);
    const auto orb = cv::ORB::create();
    std::vector<cv::KeyPoint> kp_curr, kp_prev;
    cv::Mat desc_curr, desc_prev;

    {   // initialize keypoints for first frame
        const auto inRGB = qRGB->get<dai::ImgFrame>();
        const auto& frame = inRGB->getCvFrame();
        orb->detect(frame, kp_prev);
    }

    //-- Feature Matcher
    //-- https://docs.opencv.org/3.4/d7/dff/tutorial_feature_homography.html
    const auto matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::MatcherType::FLANNBASED);
    std::vector<std::vector<cv::DMatch>> matches;

    //-- Homography MA
    std::vector<cv::Mat> homography_history;
    homography_history.reserve(HOMOGRAPY_SMOOTHING);
    for (auto i = 0; i < HOMOGRAPY_SMOOTHING; ++i)
        homography_history.push_back(cv::Mat::eye(3, 3, CV_32F));

    while (true) {
        auto imuData = qRGB->get<dai::IMUData>();
        if (imuData) {
            auto imuPackets = imuData->packets;
            for (auto& packet : imuPackets) {
                auto& gyro = packet.gyroscope;
                state.rotationAccum[0] += gyro.x;
                state.rotationAccum[1] += gyro.y;
                state.rotationAccum[2] += gyro.z;
                // std::cout << "Gyroscope: (";
                // std::cout << "x: " << gyro.x << ", ";
                // std::cout << "y: " << gyro.y << ", ";
                // std::cout << "z: " << gyro.z << ") ";
                // std::cout << "Accuracy: " << (int)gyro.accuracy << '\n';

                auto& accel = packet.acceleroMeter;
                state.translationAccum[0] += accel.x;
                state.translationAccum[1] += accel.y;
                state.translationAccum[2] += accel.z;
                // std::cout << "Accelerometer: (";
                // std::cout << "x: " << accel.x << ", ";
                // std::cout << "y: " << accel.y << ", ";
                // std::cout << "z: " << accel.z << ") ";
                // std::cout << "Accuracy: " << (int)accel.accuracy << '\n';
            }
        }

        const auto inRGB = qRGB->get<dai::ImgFrame>();
        if (inRGB) {
            matches.clear();
            const auto& frame = inRGB->getCvFrame();

            orb->detectAndCompute(frame, cv::noArray(), kp_curr, desc_curr);
            cv::drawKeypoints(frame, kp_curr, frame, blue);

            if (desc_prev.rows > MIN_MATCHES && desc_curr.rows > MIN_MATCHES) {
                if (desc_prev.type() != CV_32F)
                    desc_prev.convertTo(desc_prev, CV_32F);
                if (desc_curr.type() != CV_32F)
                    desc_curr.convertTo(desc_curr, CV_32F);

                matcher->knnMatch(desc_prev, desc_curr, matches, 2);

                if (matches.size() >= MIN_MATCHES) {
                    const float ratio_lowe = 0.75f;
                    std::vector<cv::Point2f> points_prev, points_curr;
                    for (auto i = 0; i < matches.size(); ++i) {
                        // distance is the distance between the a point in desc_1 and it's NN in desc_2
                        // per this tutorial: https://docs.opencv.org/3.4/d7/dff/tutorial_feature_homography.html
                        // This is Lowes ratio test.
                        // TODO : https://stackoverflow.com/questions/51197091/how-does-the-lowes-ratio-test-work`
                        if (matches[i][0].distance < ratio_lowe * matches[i][1].distance) {
                            points_prev.push_back(kp_prev[ matches[i][0].queryIdx ].pt);
                            points_curr.push_back(kp_curr[ matches[i][0].trainIdx ].pt);
                        }
                    }

                    if (points_prev.size() >= MIN_HOMOGRAPHY_POINTS && points_curr.size() >= MIN_HOMOGRAPHY_POINTS) {
                        auto homography = cv::findHomography(points_prev, points_curr, cv::RANSAC);
                        homography.convertTo(homography, CV_32F);

                        // TODO: Ring Buffer
                        if (homography_history.size() == HOMOGRAPY_SMOOTHING)
                            homography_history.erase(homography_history.begin());
                        homography_history.push_back(homography);

                        auto homography_avg = *homography_history.begin();
                        for (auto h = homography_history.begin() + 1; h < homography_history.end(); ++h) {
                            cv::add(homography_avg, *h, homography_avg);
                        }
                        cv::divide(homography_avg, HOMOGRAPY_SMOOTHING, homography_avg);

                        // {   // update state
                        //     std::vector<cv::Mat> rotations;
                        //     std::vector<cv::Mat> translations;
                        //     std::vector<cv::Mat> normals;
                        //     cv::decomposeHomographyMat(homography_avg, state.camIntrinsics, rotations, translations, normals); 

                        //     cv::Vec3f rot;
                        //     cv::Rodrigues(rotations[0], rot);
                        //     cv::add(state.rotation, rot, state.rotation);

                        //     cv::Vec3f rotError;
                        //     cv::subtract(state.rotation, rot, rotError);
                        //     cv::subtract(rotError, state.rotationAccum, rotError);

                        //     std::cout << "Rotation Error: (";
                        //     std::cout << "x: " << rotError[0] << ", ";
                        //     std::cout << "y: " << rotError[1] << ", ";
                        //     std::cout << "z: " << rotError[2] << ")\n";

                        //     cv::Vec3f trans;
                        //     cv::Affine3f tmat(translations[0]);
                        //     cv::add(state.translation, tmat.translation(), state.translation);

                        //     cv::Vec3f transError;
                        //     cv::subtract(state.translation, trans, transError);
                        //     cv::subtract(transError, state.translationAccum, transError);

                        //     std::cout << "Translation Error: (";
                        //     std::cout << "x: " << transError[0] << ", ";
                        //     std::cout << "y: " << transError[1] << ", ";
                        //     std::cout << "z: " << transError[2] << ")\n";

                        //     state.translationAccum.zeros();
                        //     state.rotationAccum.zeros();
                        // }

                        cv::Mat frame_warped;
                        cv::warpPerspective(frame, frame_warped, homography_avg, frame.size());

                        cv::Mat frame_compared;
                        cv::hconcat(frame, frame_warped, frame_compared);

                        cv::imshow("Warped image comparison", frame_compared);
                    }
                }
            }

            // cv::imshow("rgb", frame);

            std::swap(kp_curr, kp_prev);
            std::swap(desc_curr, desc_prev);
        }

        int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q') {
            break;
        }
    }

    return 0;
} 