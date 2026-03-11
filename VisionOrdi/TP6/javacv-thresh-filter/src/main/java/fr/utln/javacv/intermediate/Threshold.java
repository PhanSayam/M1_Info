package fr.utln.javacv.intermediate;

import static org.bytedeco.opencv.global.opencv_core.CV_8UC1;

import static org.bytedeco.opencv.global.opencv_imgcodecs.IMREAD_GRAYSCALE;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

import org.bytedeco.opencv.opencv_core.Mat;

/**
 * A class dedicated to intermediate JavaCV image thresholding.
 * @author Julien Seinturier - Universit&eacute; de Toulon / LIS umr CNRS 7020 - <a href="http://web.seinturier.fr">http://web.seinturier.fr</a>
 */
public class Threshold {

//    /**
//     * The main method.
//     * @param args the main method arguments
//     */
//    public static void main(String[] args) {
//
//        System.out.println("Exercise 1: ");
//        ex1();
//        System.out.println("");
//    }

    /**
     * The main method.
     * @param args the main method arguments
     */
    public static void main(String[] args) {

        System.out.println("Exercise 1: ");
        ex1();
        System.out.println("");

        System.out.println("Exercise 2: ");
        ex2();
        System.out.println("");

        System.out.println("Exercise 3: ");
        ex3();
        System.out.println("");

        System.out.println("Exercise 4: ");
        ex4();
        System.out.println("");

        System.out.println("Exercise 5: ");
        ex5();
        System.out.println("");

        System.out.println("Exercise 6: ");
        ex6();
        System.out.println("");
    }

    /**
     * Compute the binary threshold for the <code>src</code> image.
     * @param src the input image (grayscaledd)
     * @param tresh the threshold value
     * @param maxval the max value (for acceptec pixels)
     * @return a Mat that represents the thresholded image.
     */
    public static Mat threshBinary(Mat src, double tresh, double maxval) {
        if (src != null) {

            Mat dst = new Mat(src.rows(), src.cols(), CV_8UC1);

            threshold(src, dst, tresh, maxval, CV_THRESH_BINARY);

            return dst;
        } else {
            return new Mat();
        }
    }

    /**
     * The exercise 1 code.
     */
    public static void ex1() {
        String file = "data/images/thresh_original.bmp";

        // Load an image and converting it in grayscale
        Mat image = imread(file, IMREAD_GRAYSCALE);

        // Apply binary thresholding with tresh = 172 and maxval = 255
        Mat m = threshBinary(image, 172, 255);

        // Write the thresholded image as ex1_treshold_binary.png file
        String output = "output/ex1_treshold_binary.png";

        if (imwrite(output, m)) {
            System.out.println("wrote tresholded image to "+output+".");
        } else {
            System.out.println("Cannot save thresholded image.");
        }
    }

    /**
     * Compute the binary threshold for the <code>src</code> image.
     * @param src the input image (grayscaledd)
     * @param tresh the threshold value
     * @param maxval the max value (for rejected pixels)
     * @return a Mat that represents the thresholded image.
     */
    public static Mat threshBinaryInverted(Mat src, double tresh, double maxval) {
        if (src != null) {

            Mat dst = new Mat(src.rows(), src.cols(), CV_8UC1);

            threshold(src, dst, tresh, maxval, CV_THRESH_BINARY_INV);

            return dst;
        } else {
            return new Mat();
        }
    }

    /**
     * The exercise 2 code.
     */
    public static void ex2() {
        String file = "data/images/thresh_original.bmp";

        // Load an image and converting it in grayscale
        Mat image = imread(file, IMREAD_GRAYSCALE);

        Mat m = threshBinaryInverted(image, 172, 255);

        String output = "output/ex2_treshold_binary_inv.png";

        if (imwrite(output, m)) {
            System.out.println("wrote tresholded image to "+output+".");
        } else {
            System.out.println("Cannot save thresholded image.");
        }
    }

    /**
     * Compute the truncated threshold for the <code>src</code> image.
     * @param src the input image (grayscaledd)
     * @param tresh the threshold value
     * @return a Mat that represents the thresholded image.
     */
    public static Mat threshTruncate(Mat src, double tresh) {
        if (src != null) {

            Mat dst = new Mat(src.rows(), src.cols(), CV_8UC1);

            threshold(src, dst, tresh, tresh, CV_THRESH_TRUNC);

            return dst;
        } else {
            return new Mat();
        }
    }

    /**
     * The exercise 3 code.
     */
    public static void ex3() {
        String file = "data/images/thresh_original.bmp";

        // Load an image and converting it in grayscale
        Mat image = imread(file, IMREAD_GRAYSCALE);

        Mat m = threshTruncate(image, 120);

        String output = "output/ex3_treshold_truncate.png";

        if (imwrite(output, m)) {
            System.out.println("wrote tresholded image to "+output+".");
        } else {
            System.out.println("Cannot save thresholded image.");
        }
    }

    /**
     * Compute the threshold to zero for the <code>src</code> image.
     * @param src the input image (grayscaledd)
     * @param tresh the threshold value
     * @return a Mat that represents the thresholded image.
     */
    public static Mat threshToZero(Mat src, double tresh) {
        if (src != null) {

            Mat dst = new Mat(src.rows(), src.cols(), CV_8UC1);

            threshold(src, dst, tresh, tresh, CV_THRESH_TOZERO);

            return dst;
        } else {
            return new Mat();
        }
    }

    /**
     * The exercise 4 code.
     */
    public static void ex4() {
        String file = "data/images/thresh_original.bmp";

        // Load an image and converting it in grayscale
        Mat image = imread(file, IMREAD_GRAYSCALE);

        Mat m = threshToZero(image, 160);

        String output = "output/ex4_treshold_to_zero.png";

        if (imwrite(output, m)) {
            System.out.println("wrote tresholded image to "+output+".");
        } else {
            System.out.println("Cannot save thresholded image.");
        }
    }

    /**
     * Compute the threshold to zero for the <code>src</code> image.
     * @param src the input image (grayscaledd)
     * @param tresh the threshold value
     * @return a Mat that represents the thresholded image.
     */
    public static Mat threshToZeroInverted(Mat src, double tresh) {
        if (src != null) {

            Mat dst = new Mat(src.rows(), src.cols(), CV_8UC1);

            threshold(src, dst, tresh, tresh, CV_THRESH_TOZERO_INV);

            return dst;
        } else {
            return new Mat();
        }
    }

    /**
     * The exercise 5 code.
     */
    public static void ex5() {
        String file = "data/images/thresh_original.bmp";

        // Load an image and converting it in grayscale
        Mat image = imread(file, IMREAD_GRAYSCALE);

        Mat m = threshToZeroInverted(image, 160);

        String output = "output/ex5_treshold_to_zero_inv.png";

        if (imwrite(output, m)) {
            System.out.println("wrote tresholded image to "+output+".");
        } else {
            System.out.println("Cannot save thresholded image.");
        }
    }

    /**
     * The exercise 6 code.
     */
    public static void ex6() {

        String file = "data/images/thresh_original.bmp";

        // Load an image and converting it in grayscale
        Mat src = imread(file, IMREAD_GRAYSCALE);
    }


	
}
