package fr.utln.javacv.intermediate;

import fr.utln.javacv.jfx.JavaCVJFXImageDisplay;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;

import static java.util.stream.IntStream.range;
import static org.bytedeco.opencv.global.opencv_core.split;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import static org.bytedeco.opencv.global.opencv_core.transpose;
import static org.bytedeco.opencv.global.opencv_core.flip;

/**
 * A class dedicated to intermediate JavaCV image manipulation.
 * @author Julien Seinturier - Universit&eacute; de Toulon / LIS umr CNRS 7020 - <a href="http://web.seinturier.fr">http://web.seinturier.fr</a>
 */
public class ImageModifying {
    /**
     * Extract a sub-Mat located within a rectangle from the given <code>source</code>
     * @param source the source MAt
     * @param x the X coordinate of the rectangle upper left corner
     * @param y the y coordinate of the rectangle upper left corner
     * @param width the rectangle width
     * @param height the rectangle height
     * @return the extracted sut Mat
     */
    public static Mat extract(Mat source, int x, int y, int width, int height){
        return source.apply(new Rect(x, y, width, height));
    }

    /**
     * Split the <code>source</code> Mat into an array <code>m</code> of Mat where each <code>m[i]</code> contains values from source channel <code>i</code>.
     * @param source the source Mat
     * @return an array of Mat
     */
    public static Mat[] splitChannels(Mat source){
        Mat[] m = new Mat[source.channels()];
        for (int i=0; i < source.channels(); i++){
            m[i] = new Mat();
            m[i].create(source.size(), source.depth());
        }
        return m;
    }

    public static void main(String[] args) {
        String file = "data/images/lake-mountain.png";

        // Load an image, using its color model
        Mat image = imread(file, IMREAD_UNCHANGED);

        Mat m = extract(image, 200, 250, 100, 100);

        JavaCVJFXImageDisplay.display("Extract", m);

    }

//    public static void main(String[] args) {
//        String file = "data/images/lake-mountain.png";
//
//        // Load an image, using its color model
//        Mat image = imread(file, IMREAD_UNCHANGED);
//
//        Mat[] splitted = splitChannels(image);
//
//        if (splitted != null){
//            for(int i = 0; i < splitted.length; i++){
//                imwrite("splitted_"+i+".png", splitted[i]);
//            }
//        }
//
//    }

    /**
     * Return a new {@link Mat} that is the result of a right rotation
     * of the given <code>source</code> {@link Mat} by 90°.
     * @param source the source Mat
     * @return the rotated Mat
     */
    public static Mat rot90(Mat source){
        Mat dest = new Mat();
        transpose(source, dest);
        flip(dest, dest, 1);
        return dest;
    }

//    public static void main(String[] args) {
//        String file = "data/images/lake-mountain.png";
//
//        // Load an image, using its color model
//        Mat image = imread(file, IMREAD_UNCHANGED);
//
//        Mat m = rot90(image);
//
//        imwrite("Rot90.png", m);
//
//    }
}
