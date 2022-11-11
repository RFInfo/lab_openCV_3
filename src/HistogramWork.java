import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import java.util.ArrayList;
import java.util.List;

public class HistogramWork {

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        Mat src = new Mat();
        Mat grayFrame = new Mat();
        Mat luminosityHistImage = new Mat(new Size(512, 200), CvType.CV_8UC3);
        Mat bgrHistImage = new Mat(new Size(512, 200), CvType.CV_8UC3);

        Mat dstHistImage = new Mat(new Size(512, 400), CvType.CV_8UC3);
        Mat roi1 = dstHistImage.submat(new Rect(0,0,512,200));
        Mat roi2 = dstHistImage.submat(new Rect(0,200,512,200));

        VideoCapture capture = new VideoCapture(0);

        if(capture.isOpened()){
        int key = 0;
            while(key != 27) {
                if (!capture.read(src)) break;

                Imgproc.cvtColor(src,grayFrame, Imgproc.COLOR_BGR2GRAY);

                histogramLuminosity(grayFrame, luminosityHistImage);
                histogramBGR(src, bgrHistImage);

                luminosityHistImage.copyTo(roi1);
                bgrHistImage.copyTo(roi2);

                HighGui.imshow("Src", src);

                HighGui.imshow("BGR HistImage", bgrHistImage);
                HighGui.imshow("Luminosity HistImage", luminosityHistImage);
                HighGui.moveWindow("BGR HistImage",src.width(), 0);
                HighGui.moveWindow("Luminosity HistImage",src.width(), 300);

                HighGui.imshow("Dst HistImage", dstHistImage);
                HighGui.moveWindow("Dst HistImage",src.width(), 0);

                key = HighGui.waitKey(20);
            }

        }

        HighGui.destroyAllWindows();
        System.exit(0);
    }

    public static Mat histogramLuminosity(Mat src, Mat histImage){
        List<Mat> histogramSource = new ArrayList<>();
        histogramSource.add(src);

        int histSize = 256;
        int binSize = histImage.width()/histSize;
        float[] ranges = {0,256};
        Mat histData = new Mat();
        Imgproc.calcHist(histogramSource,new MatOfInt(0),new Mat(),histData,new MatOfInt(histSize),
                new MatOfFloat(ranges),false);

        Core.normalize(histData,histData,0,histImage.rows(),Core.NORM_MINMAX);

//        System.out.println(histogramData.dump());
        float[] data = new float[(int)histData.total() * histData.channels()];
        histData.get(0,0,data);

        Imgproc.rectangle(histImage,new Rect(0, 0, histImage.width(), histImage.height()), new Scalar(0),Imgproc.FILLED);
        for (int i = 1; i < histSize ; i++) {
            Imgproc.line(histImage,new Point(binSize*(i-1),histImage.height()-data[i-1]), new Point(binSize*i,histImage.height()-data[i]),
                    new Scalar(255,255,255), 2);
        }

        return histImage;
    }

    public static Mat histogramBGR(Mat src, Mat histImage){
        List<Mat> histogramSource = new ArrayList<>();
        Core.split(src, histogramSource);


        int histSize = 256;
        int binSize = histImage.width()/histSize;
        float[] ranges = {0,256};
        Mat histDataB = new Mat();
        Mat histDataG = new Mat();
        Mat histDataR = new Mat();

        Imgproc.calcHist(histogramSource,new MatOfInt(0),new Mat(),histDataB,new MatOfInt(histSize),
                new MatOfFloat(ranges),false);
        Imgproc.calcHist(histogramSource,new MatOfInt(1),new Mat(),histDataG,new MatOfInt(histSize),
                new MatOfFloat(ranges),false);
        Imgproc.calcHist(histogramSource,new MatOfInt(2),new Mat(),histDataR,new MatOfInt(histSize),
                new MatOfFloat(ranges),false);

        Core.normalize(histDataB,histDataB,0,histImage.rows(),Core.NORM_MINMAX);
        Core.normalize(histDataG,histDataG,0,histImage.rows(),Core.NORM_MINMAX);
        Core.normalize(histDataR,histDataR,0,histImage.rows(),Core.NORM_MINMAX);

//        System.out.println(histogramData.dump());
        float[] dataB = new float[(int)histDataB.total() * histDataB.channels()];
        float[] dataG = new float[(int)histDataG.total() * histDataG.channels()];
        float[] dataR = new float[(int)histDataR.total() * histDataR.channels()];
        histDataB.get(0,0,dataB);
        histDataG.get(0,0,dataG);
        histDataR.get(0,0,dataR);

        Imgproc.rectangle(histImage,new Rect(0, 0, histImage.width(), histImage.height()), new Scalar(0),Imgproc.FILLED);
        for (int i = 1; i < histSize ; i++) {
            Imgproc.line(histImage,new Point(binSize*(i-1),histImage.height()-dataB[i-1]), new Point(binSize*i,histImage.height()-dataB[i]),
                    new Scalar(255,0,0), 2);
            Imgproc.line(histImage,new Point(binSize*(i-1),histImage.height()-dataG[i-1]), new Point(binSize*i,histImage.height()-dataG[i]),
                    new Scalar(0,255,0), 2);
            Imgproc.line(histImage,new Point(binSize*(i-1),histImage.height()-dataR[i-1]), new Point(binSize*i,histImage.height()-dataR[i]),
                    new Scalar(0,0,255), 2);
        }

        return histImage;
    }
}
