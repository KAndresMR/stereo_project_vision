#include "DisparityMode.hpp"
#include "Rectifier.hpp"
#include "SGBMProcessor.hpp"
#include "Kalman1D.hpp"
#include "DiagnosticLogger.hpp"
#include "DebugViews.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>

// ─────────────────────────────────────────────────────────────────────────────
// getMedianDisparity — median disparity in a ROI (CV_16S, returns px units)
// ─────────────────────────────────────────────────────────────────────────────
static float getMedianDisparity(const cv::Mat& disp16S, cv::Rect roi) {
    roi &= cv::Rect(0, 0, disp16S.cols, disp16S.rows);
    cv::Mat crop = disp16S(roi);

    std::vector<float> vals;
    vals.reserve(crop.total());
    for (int y = 0; y < crop.rows; ++y)
        for (int x = 0; x < crop.cols; ++x)
            if (short d = crop.at<short>(y, x); d > 0)
                vals.push_back(d / 16.0f);

    if (vals.empty()) return -1.0f;
    std::sort(vals.begin(), vals.end());
    return vals[vals.size() / 2];
}

// ─────────────────────────────────────────────────────────────────────────────
// buildDashboardTile — resize + convert to BGR for the dashboard grid
// ─────────────────────────────────────────────────────────────────────────────
static cv::Mat buildDashboardTile(const cv::Mat& img, int w, int h) {
    if (img.empty()) return cv::Mat::zeros(h, w, CV_8UC3);
    cv::Mat out;
    if (img.channels() == 1) cv::cvtColor(img, out, cv::COLOR_GRAY2BGR);
    else                     out = img.clone();
    cv::resize(out, out, cv::Size(w, h));
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// runLiveDisparity — main SGBM pipeline with AR overlay and dashboard
// ─────────────────────────────────────────────────────────────────────────────
void runLiveDisparity(CameraStream& cam1, CameraStream& cam2,
                      const CalibrationConfig& config) {
    Rectifier rectifier(config);
    if (!rectifier.compute()) { 
        Log::error("DisparityMode", "Rectification init failed"); 
        return; 
    }

    // ── SGBM initial params ───────────────────────────────────────────────────
    SGBMProcessor::Params params;
    params.numDisparities    = 64;
    params.blockSize         = 7;
    params.uniquenessRatio   = 10;
    params.speckleWindowSize = 100;
    params.speckleRange      = 2;
    SGBMProcessor sgbm(params);

    // ── Dashboard window + trackbars ──────────────────────────────────────────
    const std::string winName = "Stereo Vision Dashboard";
    cv::namedWindow(winName, cv::WINDOW_AUTOSIZE);

    int tbNumDisp    = 8;   // *16 → 64
    int tbBlockSize  = 9;
    int tbUniqueness = 25;
    int tbSpeckleWin = 200;
    int tbClaheClip  = 25;  // /10 → 2.5
    int tbClaheTile  = 8;
    int tbPreBlur    = 8;   // /10 → 0.8

    cv::createTrackbar("numDisp x16", winName, &tbNumDisp,    20);
    cv::createTrackbar("blockSize",   winName, &tbBlockSize,  21);
    cv::createTrackbar("uniqueness",  winName, &tbUniqueness, 50);
    cv::createTrackbar("speckleWin",  winName, &tbSpeckleWin, 200);
    cv::createTrackbar("CLAHE Clip",  winName, &tbClaheClip,  100);
    cv::createTrackbar("CLAHE Tile",  winName, &tbClaheTile,  32);
    cv::createTrackbar("Pre-Blur",    winName, &tbPreBlur,    30);

    std::cout << "\n  Live SGBM mode — adjust sliders — ESC = menu\n\n";

    // ── Kalman filters (one per measurement window) ───────────────────────────
    Kalman1D kalmanFijo(1e-3f, 0.1f);      // centre cross-hair
    Kalman1D kalmanDinamico(1e-3f, 0.1f);  // AR target tracker

    // Physical constants from stereo calibration
    const float FOCAL_PX   = 600.0f;   // avg of fx=600.6 fy=599.8
    const float BASELINE_M = 0.07414f; // 74.14 mm

    SGBMProcessor::Params lastParams = params;

    while (true) {
        // ── Read trackbars → update params ────────────────────────────────────
        params.numDisparities    = std::max(16, tbNumDisp * 16);
        params.blockSize         = std::max(3, tbBlockSize | 1); // force odd
        params.uniquenessRatio   = tbUniqueness;
        params.speckleWindowSize = tbSpeckleWin;
        params.claheClipLimit    = std::max(0.1, tbClaheClip / 10.0);
        params.claheTileSize     = std::max(2, tbClaheTile);
        params.preBlurSigma      = tbPreBlur / 10.0;

        if (params.numDisparities != lastParams.numDisparities ||
            params.blockSize      != lastParams.blockSize        ||
            params.uniquenessRatio  != lastParams.uniquenessRatio  ||
            params.speckleWindowSize!= lastParams.speckleWindowSize||
            params.claheClipLimit != lastParams.claheClipLimit   ||
            params.claheTileSize  != lastParams.claheTileSize    ||
            params.preBlurSigma   != lastParams.preBlurSigma) {
            sgbm.setParams(params);
            lastParams = params;
            Log::info("DisparityMode", "SGBM params updated");
        }

        // ── Grab frames ───────────────────────────────────────────────────────
        cv::Mat f1 = grabFrame(cam1);
        cv::Mat f2 = grabFrame(cam2);
        if (f1.empty() || f2.empty()) { cv::waitKey(1); continue; }

        // ── SGBM pipeline ─────────────────────────────────────────────────────
        auto [rL, rR]        = rectifier.rectify(f1, f2);
        cv::Mat rawDisp      = sgbm.compute(rL, rR);
        cv::Mat rawVis       = sgbm.visualize(rawDisp);
        cv::Mat claheView    = sgbm.getEnhancedLeft();
        cv::Mat wlsDisp      = sgbm.postprocess(rawDisp, rL);
        cv::Mat wlsVis       = sgbm.visualize(wlsDisp);
        cv::Mat temporalDisp = sgbm.temporalSmooth(wlsDisp);
        cv::Mat temporalVis;
        if (!temporalDisp.empty()) {
            temporalVis = sgbm.visualize(temporalDisp);
            cv::putText(temporalVis,
                "numDisp=" + std::to_string(params.numDisparities) +
                " block="  + std::to_string(params.blockSize),
                {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {255,255,255}, 2);
        }

        // ── Window 1: fixed centre depth measurement ──────────────────────────
        {
            cv::Mat view      = rL.clone();
            cv::Point centre  = {view.cols / 2, view.rows / 2};
            cv::Rect  roiRect = {centre.x - 10, centre.y - 10, 20, 20};
            float disp        = getMedianDisparity(rawDisp, roiRect);
            float z           = -1.0f;
            if (disp > 0.0f) {
                z = kalmanFijo.update((FOCAL_PX * BASELINE_M / disp) * 100.0f);
                if (z > 0.0f && z < 400.0f) {
                    cv::circle(view, centre, 5, {0,255,0}, -1);
                    std::ostringstream ss;
                    ss << std::fixed << std::setprecision(2) << "Z: " << z << " cm";
                    cv::putText(view, ss.str(), {centre.x + 15, centre.y},
                                cv::FONT_HERSHEY_SIMPLEX, 0.7, {0,255,0}, 2);
                }
            } else {
                kalmanFijo.update(-1.0f);
            }
            cv::imshow("1 - Medicion de Profundidad", view);
        }

        // ── Window 2: AR dynamic target tracker ──────────────────────────────
        {
            cv::Mat arFrame = rL.clone();
            double  minD, maxD;
            cv::minMaxLoc(temporalDisp, &minD, &maxD);

            cv::Point targetPoint = {arFrame.cols / 2, arFrame.rows / 2};
            bool targetFound      = false;
            std::vector<std::vector<cv::Point>> contours;
            int largestIdx = -1;

            if (maxD > 16.0) {
                cv::Mat mask = (temporalDisp > (maxD - 80));
                mask.convertTo(mask, CV_8U, 1);
                cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, {9,9});
                cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
                cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

                double maxArea = 0, minAreaTh = 400, maxAreaTh = arFrame.total() * 0.40;
                for (size_t i = 0; i < contours.size(); ++i) {
                    double a = cv::contourArea(contours[i]);
                    if (a > maxArea && a > minAreaTh && a < maxAreaTh) { maxArea = a; largestIdx = (int)i; }
                }
                if (largestIdx >= 0) {
                    targetFound = true;
                    cv::Moments m = cv::moments(contours[largestIdx]);
                    targetPoint  = {int(m.m10 / m.m00), int(m.m01 / m.m00)};
                }
            }

            float zDyn = -1.0f;
            if (targetFound) {
                cv::Rect tr = {targetPoint.x - 10, targetPoint.y - 10, 20, 20};
                float d = getMedianDisparity(rawDisp, tr);
                zDyn = kalmanDinamico.update(d > 0.0f ? (FOCAL_PX * BASELINE_M / d) * 100.0f : -1.0f);
            } else {
                kalmanDinamico.update(-1.0f);
            }

            if (targetFound && zDyn > 0.0f && zDyn < 150.0f) {
                float t   = std::clamp((zDyn - 30.0f) / 120.0f, 0.0f, 1.0f);
                cv::Scalar col(0, int(255*t), int(255*(1-t)));
                cv::drawContours(arFrame, contours, largestIdx, col, 3, cv::LINE_AA);
                cv::Rect bbox = cv::boundingRect(contours[largestIdx]);
                const int L   = 20;
                // Sci-fi corner brackets
                cv::line(arFrame, {bbox.x,            bbox.y},            {bbox.x+L,         bbox.y},            col, 2);
                cv::line(arFrame, {bbox.x,            bbox.y},            {bbox.x,            bbox.y+L},          col, 2);
                cv::line(arFrame, {bbox.x+bbox.width, bbox.y},            {bbox.x+bbox.width-L,bbox.y},           col, 2);
                cv::line(arFrame, {bbox.x+bbox.width, bbox.y},            {bbox.x+bbox.width, bbox.y+L},          col, 2);
                cv::line(arFrame, {bbox.x,            bbox.y+bbox.height},{bbox.x+L,          bbox.y+bbox.height},col, 2);
                cv::line(arFrame, {bbox.x,            bbox.y+bbox.height},{bbox.x,            bbox.y+bbox.height-L},col,2);
                cv::line(arFrame, {bbox.x+bbox.width, bbox.y+bbox.height},{bbox.x+bbox.width-L,bbox.y+bbox.height},col,2);
                cv::line(arFrame, {bbox.x+bbox.width, bbox.y+bbox.height},{bbox.x+bbox.width, bbox.y+bbox.height-L},col,2);
                cv::circle(arFrame, targetPoint, 4, col, -1, cv::LINE_AA);
                std::ostringstream ds;
                ds << std::fixed << std::setprecision(1) << "Z: " << zDyn << " cm";
                cv::putText(arFrame, "TARGET LOCKED", {bbox.x, bbox.y-20},
                            cv::FONT_HERSHEY_DUPLEX, 0.5, col, 1, cv::LINE_AA);
                cv::putText(arFrame, ds.str(), {bbox.x, bbox.y-5},
                            cv::FONT_HERSHEY_DUPLEX, 0.5, col, 1, cv::LINE_AA);
            } else {
                cv::Point sc  = {arFrame.cols/2, arFrame.rows/2};
                cv::Scalar gc = {150,150,150};
                cv::circle(arFrame, sc, 60, gc, 1, cv::LINE_AA);
                static double angle = 0.0; angle += 0.1;
                cv::line(arFrame, sc, {sc.x + int(60*std::cos(angle)), sc.y + int(60*std::sin(angle))}, gc, 1);
                cv::putText(arFrame, "SEARCHING...", {sc.x-50, sc.y-75},
                            cv::FONT_HERSHEY_DUPLEX, 0.5, gc, 1, cv::LINE_AA);
            }
            cv::imshow("2 - Efecto Realidad Aumentada", arFrame);
        }

        // ── Dashboard grid ────────────────────────────────────────────────────
        {
            cv::Mat stereoView;
            cv::hconcat(rL, rR, stereoView);
            for (int y = 0; y < stereoView.rows; y += 40)
                cv::line(stereoView, {0,y}, {stereoView.cols,y}, {0,255,0}, 1);

            cv::Mat t1 = buildDashboardTile(claheView,   320, 240); DebugViews::label(t1, "00 - CLAHE");
            cv::Mat t2 = buildDashboardTile(rawVis,      320, 240); DebugViews::label(t2, "01 - RAW SGBM");
            cv::Mat t3 = buildDashboardTile(wlsVis,      320, 240); DebugViews::label(t3, "02 - WLS FILTER");
            cv::Mat t4 = buildDashboardTile(stereoView,  640, 240); DebugViews::label(t4, "RECTIFIED PAIR (Epipolar Check)");
            cv::Mat t5 = buildDashboardTile(temporalVis, 320, 240); DebugViews::label(t5, "03 - TEMPORAL (FINAL)");

            cv::Mat topRow, bottomRow, dashboard;
            cv::hconcat(std::vector<cv::Mat>{t1, t2, t3}, topRow);
            cv::hconcat(t4, t5, bottomRow);
            cv::vconcat(topRow, bottomRow, dashboard);
            cv::imshow(winName, dashboard);
        }

        if (cv::waitKey(1) == 27) break;
    }
    cv::destroyWindow(winName);
}
