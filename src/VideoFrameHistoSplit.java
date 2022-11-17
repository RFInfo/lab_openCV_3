import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

import java.util.ArrayList;
import java.util.List;

public class VideoFrameHistoSplit {

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {
        int histW = 256;
        int histH = 100;

        Mat src = new Mat();
        Mat dst = new Mat();

        Mat logo = Imgcodecs.imread("./test_images/logo.png", Imgcodecs.IMREAD_UNCHANGED);

        VideoCapture cam = new VideoCapture(0);
//        VideoCapture cam = new VideoCapture("http://192.168.100.8:8080/video");

        histH = (int)(cam.get(Videoio.CAP_PROP_FRAME_HEIGHT)/4);

        if (!cam.isOpened()) return;
        while (true) {
            cam.read(src);

            if (src.empty()) break;

            // create and destination frame with sidebar for histograms
            dst = new Mat(new Size(src.width() + histW, src.height()), CvType.CV_8UC3);

            // get 4 ROIs from source frame
            Mat srcROI1 = src.submat(new Rect(0, 0, src.width() / 2, src.height() / 2));
            Mat srcROI2 = src.submat(new Rect(src.width() / 2, 0, src.width() / 2, src.height() / 2));
            Mat srcROI3 = src.submat(new Rect(0, src.height() / 2, src.width() / 2, src.height() / 2));
            Mat srcROI4 = src.submat(new Rect(src.width() / 2, src.height() / 2, src.width() / 2, src.height() / 2));

            // destination regions for source frames
            Mat dstROI0 = dst.submat(new Rect(0, 0, src.width(), src.height()));

            // destinations regions for displaying 4 histograms
            Mat dstROI1 = dst.submat(new Rect(dstROI0.width(), histH * 0, histW, histH));
            Mat dstROI2 = dst.submat(new Rect(dstROI0.width(), histH * 1, histW, histH));
            Mat dstROI3 = dst.submat(new Rect(dstROI0.width(), histH * 2, histW, histH));
            Mat dstROI4 = dst.submat(new Rect(dstROI0.width(), histH * 3, histW, histH));

//                HighGui.imshow("Src1",roi1);
//                HighGui.imshow("Src2",roi2);
//                HighGui.imshow("Src3",roi3);
//                HighGui.imshow("Src4",roi4);

            // copy src in dst frame assembly
            Core.copyTo(src, dstROI0, new Mat());

            // draw roi separation lines
            Imgproc.line(dstROI0, new Point(0, (dstROI0.height() - 1) / 2), new Point(dstROI0.width() - 1, (dstROI0.height() - 1) / 2), new Scalar(0, 255, 0));
            Imgproc.line(dstROI0, new Point((dstROI0.width() - 1) / 2, 0), new Point((dstROI0.width() - 1) / 2, dstROI0.height() - 1), new Scalar(0, 255, 0));

            // generate histograms and on sidebar ROIs
            histogramBGR(srcROI1, dstROI1);
            histogramBGR(srcROI2, dstROI2);
            histogramBGR(srcROI3, dstROI3);
            histogramBGR(srcROI4, dstROI4);

            // add simple logo to a frame
//            addLogo(src, logo);
            addLogo(dstROI0, logo);

            HighGui.imshow("Dst", dst);
//            HighGui.imshow("Src", src);

            int key = HighGui.waitKey(20);
            if (key == 27) {
                HighGui.destroyAllWindows();
                System.exit(0);
            }
        }
    }

    public static void addLogo(Mat src, Mat logo) {
        Mat roi = src.submat(new Rect(new Point(0, 0), logo.size()));

        Mat mask = new Mat();
        Core.extractChannel(logo, mask, 0);

//                Core.bitwise_not(logo,logo);
        Core.bitwise_not(mask, mask);
        Mat t = new Mat(mask.size(), CvType.CV_8UC3, new Scalar(255, 255, 255));

        Core.add(roi, t, roi, mask);
//                mask.copyTo(roi,mask);
    }

    public static Mat histogramBGR(Mat src, Mat histImage) {
        List<Mat> histogramSource = new ArrayList<>();
        Core.split(src, histogramSource);

        int histSize = histImage.width();
        int binSize = histImage.width() / histSize;
        float[] ranges = {0, 256};
        Mat histDataB = new Mat();
        Mat histDataG = new Mat();
        Mat histDataR = new Mat();

        Imgproc.calcHist(histogramSource, new MatOfInt(0), new Mat(), histDataB, new MatOfInt(histSize),
                new MatOfFloat(ranges), false);
        Imgproc.calcHist(histogramSource, new MatOfInt(1), new Mat(), histDataG, new MatOfInt(histSize),
                new MatOfFloat(ranges), false);
        Imgproc.calcHist(histogramSource, new MatOfInt(2), new Mat(), histDataR, new MatOfInt(histSize),
                new MatOfFloat(ranges), false);

        Core.normalize(histDataB, histDataB, 0, histImage.rows(), Core.NORM_MINMAX);
        Core.normalize(histDataG, histDataG, 0, histImage.rows(), Core.NORM_MINMAX);
        Core.normalize(histDataR, histDataR, 0, histImage.rows(), Core.NORM_MINMAX);

//        System.out.println(histogramData.dump());
        float[] dataB = new float[(int) histDataB.total() * histDataB.channels()];
        float[] dataG = new float[(int) histDataG.total() * histDataG.channels()];
        float[] dataR = new float[(int) histDataR.total() * histDataR.channels()];
        histDataB.get(0, 0, dataB);
        histDataG.get(0, 0, dataG);
        histDataR.get(0, 0, dataR);

        Imgproc.rectangle(histImage, new Rect(0, 0, histImage.width(), histImage.height()), new Scalar(0), Imgproc.FILLED);
        for (int i = 1; i < histSize; i++) {
            Imgproc.line(histImage, new Point(binSize * (i - 1), histImage.height() - dataB[i - 1]), new Point(binSize * i, histImage.height() - dataB[i]),
                    new Scalar(255, 0, 0), 2);
            Imgproc.line(histImage, new Point(binSize * (i - 1), histImage.height() - dataG[i - 1]), new Point(binSize * i, histImage.height() - dataG[i]),
                    new Scalar(0, 255, 0), 2);
            Imgproc.line(histImage, new Point(binSize * (i - 1), histImage.height() - dataR[i - 1]), new Point(binSize * i, histImage.height() - dataR[i]),
                    new Scalar(0, 0, 255), 2);
        }

        return histImage;
    }
}
