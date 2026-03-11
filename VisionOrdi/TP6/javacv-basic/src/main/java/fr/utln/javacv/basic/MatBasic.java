package fr.utln.javacv.basic;

import org.bytedeco.javacpp.indexer.DoubleIndexer;
import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgcodecs.IMREAD_UNCHANGED ;
import static org.bytedeco.opencv.global.opencv_core.CV_8U;
import static org.bytedeco.opencv.global.opencv_core.CV_8S;
import static org.bytedeco.opencv.global.opencv_core.CV_16S;
import static org.bytedeco.opencv.global.opencv_core.CV_16U;
import static org.bytedeco.opencv.global.opencv_core.CV_32F;
import static org.bytedeco.opencv.global.opencv_core.CV_32S;
import static org.bytedeco.opencv.global.opencv_core.CV_64F;
import static org.bytedeco.opencv.global.opencv_core.CV_8UC1;
import static org.bytedeco.opencv.global.opencv_core.CV_8UC2;
import static org.bytedeco.opencv.global.opencv_core.CV_8UC3;
import static org.bytedeco.opencv.global.opencv_core.CV_8UC4;
import static org.bytedeco.opencv.global.opencv_core.CV_8SC1;
import static org.bytedeco.opencv.global.opencv_core.CV_8SC2;
import static org.bytedeco.opencv.global.opencv_core.CV_8SC3;
import static org.bytedeco.opencv.global.opencv_core.CV_8SC4;
import static org.bytedeco.opencv.global.opencv_core.CV_16UC1;
import static org.bytedeco.opencv.global.opencv_core.CV_16UC2;
import static org.bytedeco.opencv.global.opencv_core.CV_16UC3;
import static org.bytedeco.opencv.global.opencv_core.CV_16UC4;
import static org.bytedeco.opencv.global.opencv_core.CV_16SC1;
import static org.bytedeco.opencv.global.opencv_core.CV_16SC2;
import static org.bytedeco.opencv.global.opencv_core.CV_16SC3;
import static org.bytedeco.opencv.global.opencv_core.CV_16SC4;
import static org.bytedeco.opencv.global.opencv_core.CV_32SC1;
import static org.bytedeco.opencv.global.opencv_core.CV_32SC2;
import static org.bytedeco.opencv.global.opencv_core.CV_32SC3;
import static org.bytedeco.opencv.global.opencv_core.CV_32SC4;
import static org.bytedeco.opencv.global.opencv_core.CV_32FC1;
import static org.bytedeco.opencv.global.opencv_core.CV_32FC2;
import static org.bytedeco.opencv.global.opencv_core.CV_32FC3;
import static org.bytedeco.opencv.global.opencv_core.CV_32FC4;
import static org.bytedeco.opencv.global.opencv_core.CV_64FC1;
import static org.bytedeco.opencv.global.opencv_core.CV_64FC2;
import static org.bytedeco.opencv.global.opencv_core.CV_64FC3;
import static org.bytedeco.opencv.global.opencv_core.CV_64FC4;

import org.bytedeco.javacpp.indexer.UByteIndexer;
/**
 * A class dedicated to basic JavaCV Mat operations.
 * @author Julien Seinturier - Universit&eacute; de Toulon / LIS umr CNRS 7020 - <a href="http://web.seinturier.fr">http://web.seinturier.fr</a>
 */
public class MatBasic {

    public static void ex6() {

        String file = "data/images/lake-mountain.png";

        // Load an image, using its color model
        Mat mat = imread(file, IMREAD_UNCHANGED);

        Size s = mat.size();
        System.out.println(s.width());
        System.out.println(s.height());
        System.out.println(mat.cols());
        System.out.println(mat.rows());



    }

    /**
     * Convert a {@link Mat#depth() Mat depth} into a {@link String}.
     * If the given <code>depth</code> is not valid, "XX" is returned.
     * @param depth the depth of a Mat (obtained using {@link Mat#depth()})
     * @return the String representation of a Mat depth
     */
    public static String getDepthAsString(int depth) {
        switch(depth) {
            case CV_8U: return "8U";
            case CV_8S: return "8S";
            case CV_16S: return "16S";
            case CV_16U: return "16U";
            case CV_32F: return "32F";
            case CV_32S: return "32S";
            case CV_64F: return "64F";
            default: return "XX";
        }
    }

    public static void ex7() {

        String file = "data/images/lake-mountain.png";

        // Load an image, using its color model
        Mat m = imread(file, IMREAD_UNCHANGED);

        System.out.println("Depth: " + getDepthAsString(m.depth()));
    }

    /**
     * Convert a {@link Mat#channels() Mat channels count} into a {@link String}.
     * If the given <code>channels<code> count is not valid, "CY" is returned.
     * @param channels the number of channels for a Mat (obtained using {@link Mat#channels()})
     * @return the String representation of a Mat channels count
     */
    public static String getChannelAsString(int channels) {
        if ((channels >= 1) && (channels <= 4)) {
            return "C"+channels;
        }

        return "CY";
    }

    public static void ex8() {

        String file = "data/images/lake-mountain.png";

        // Load an image, using its color model
        Mat m = imread(file, IMREAD_UNCHANGED);

        System.out.println("Channels: " + getChannelAsString(m.channels()));
    }

    /**
     * Convert a {@link Mat#type() Mat type} into a {@link String}.
     * If the given <code>type<code> is not valid, "XXYCZ" is returned.
     * @param type the type of a Mat (obtained using {@link Mat#type()})
     * @return the String representation of a Mat channels type
     */
    public static String getTypeAsString(int type) {
        if (type == CV_8UC1)  return "8UC1";
        if (type == CV_8UC2)  return "8UC2";
        if (type == CV_8UC3)  return "8UC3";
        if (type == CV_8UC4)  return "8UC4";
        if (type == CV_8SC1)  return "8SC1";
        if (type == CV_8SC2)  return "8SC2";
        if (type == CV_8SC3)  return "8SC3";
        if (type == CV_8SC4)  return "8SC4";
        if (type == CV_16UC1) return "16UC1";
        if (type == CV_16UC2) return "16UC2";
        if (type == CV_16UC3) return "16UC3";
        if (type == CV_16UC4) return "16UC4";
        if (type == CV_16SC1) return "16SC1";
        if (type == CV_16SC2) return "16SC2";
        if (type == CV_16SC3) return "16SC3";
        if (type == CV_16SC4) return "16SC4";
        if (type == CV_32SC1) return "32SC1";
        if (type == CV_32SC2) return "32SC2";
        if (type == CV_32SC3) return "32SC3";
        if (type == CV_32SC4) return "32SC4";
        if (type == CV_32FC1) return "32FC1";
        if (type == CV_32FC2) return "32FC2";
        if (type == CV_32FC3) return "32FC3";
        if (type == CV_32FC4) return "32FC4";
        if (type == CV_64FC1) return "64FC1";
        if (type == CV_64FC2) return "64FC2";
        if (type == CV_64FC3) return "64FC3";
        if (type == CV_64FC4) return "64FC4";
        else return "XXYCZ";
    }

    //Modify the ex9 method in order to display the size, the depth, the channels and the type of m.
    public static void ex9() {

        String file = "data/images/lake-mountain.png";

        // Load an image, using its color model
        Mat m = imread(file, IMREAD_UNCHANGED);

        System.out.println("Size: " + m.size().width() + "x" + m.size().height());
        System.out.println("Depth: " + getDepthAsString(m.depth()));
        System.out.println("Channels: " + getChannelAsString(m.channels()));
        System.out.println("Type: " + getTypeAsString(m.type()));
    }

    public static void ex10() {
        Mat m = MatUtils.rowMajorMat(4, 3, CV_8UC1);

        // Create an indexer for accessing Mat data
        UByteIndexer indexer = m.createIndexer();

        for(int row = 0; row < m.rows(); row++) {
            System.out.print("[");
            for(int col = 0; col < m.cols(); col++) {

                // Display the value of the Mat cell at (row, col)
                System.out.print(" "+indexer.get(row, col));
            }

            System.out.println(" ]");
        }
    }

    public static void ex11() {

        // Create a new Mat with 5 rows, 7 cols and 3 channel made of float
        Mat m = MatUtils.rowMajorMat(5, 7, CV_32FC3);

        System.out.println("Mat size: "+m.rows()+"x"+m.cols()+", type: "+getTypeAsString(m.type()));

        // Access to the native image raster (the Mat)
        FloatIndexer indexer = m.createIndexer();

        for(int row = 0; row < m.rows(); row++) {
            System.out.print("[");
            for(int col = 0; col < m.cols(); col++) {

                System.out.print(" ( ");
                for(int channel = 0; channel < m.channels(); channel++) {
                    System.out.print(indexer.get(row, col, channel)+" ");
                }
                System.out.print(")");
            }
            System.out.println(" ]");
        }
    }

    public static void ex12() {

        // Create a new Mat with 6 rows, 6 cols and 3 channel made of float
        Mat m = MatUtils.rowMajorMat(6, 6, CV_64FC2);

        System.out.println("Mat size: "+m.rows()+"x"+m.cols()+", type: "+getTypeAsString(m.type()));

        // Access to the native image raster (the Mat)
        DoubleIndexer indexer = m.createIndexer();

        // Prepare the channel buffer
        double[] values = new double[m.channels()];

        for(int row = 0; row < m.rows(); row++) {
            System.out.print("[");
            for(int col = 0; col < m.cols(); col++) {

                // Get the channel values as an array
                indexer.get(row, col, values);

                System.out.print(" ( ");
                for(int channel = 0; channel < m.channels(); channel++) {
                    System.out.print(values[channel]+" ");
                }
                System.out.print(")");
            }
            System.out.println(" ]");
        }
    }

    public static void ex13() {

        // Create a 4x3 mat initialized to 0
        Mat m = Mat.zeros(4, 3, CV_8UC1).asMat();

        // Create an indexer for accessing Mat data
        UByteIndexer indexer = m.createIndexer();

        // Set the values of the Mat
        int i = 1;
        for(int row = 0; row < m.rows(); row++) {
            for(int col = 0; col < m.cols(); col++) {
                indexer.put(row, col, i);
                i++;
            }
        }

        // Display the modified Mat
        for(int row = 0; row < m.rows(); row++) {
            System.out.print("[");
            for(int col = 0; col < m.cols(); col++) {

                // Display the value of the Mat cell at (row, col)
                System.out.print(" "+indexer.get(row, col));
            }

            System.out.println(" ]");
        }
    }

    public static void ex14() {

        // Create a 5x7 mat initialized to 0
        Mat m = Mat.zeros(5, 7, CV_32FC3).asMat();

        // Create an indexer for accessing Mat data
        FloatIndexer indexer = m.createIndexer();

        // Set the values of the Mat
        int i = 1;
        for(int row = 0; row < m.rows(); row++) {
            for(int col = 0; col < m.cols(); col++) {
                for(int channel = 0; channel < m.channels(); channel++)
                    indexer.put(row, col, channel, i);
                i++;
            }
        }

        // Display the modified Mat
        for(int row = 0; row < m.rows(); row++) {
            System.out.print("[");
            for(int col = 0; col < m.cols(); col++) {

                System.out.print(" ( ");
                for(int channel = 0; channel < m.channels(); channel++) {
                    System.out.print(indexer.get(row, col, channel)+" ");
                }
                System.out.print(")");
            }
            System.out.println(" ]");
        }
    }

}
