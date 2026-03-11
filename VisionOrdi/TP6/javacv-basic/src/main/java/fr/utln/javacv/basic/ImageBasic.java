package fr.utln.javacv.basic;
import org.bytedeco.opencv.opencv_core.Mat;

import static fr.utln.javacv.basic.MatBasic.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgcodecs.* ;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;
import fr.utln.javacv.jfx.JavaCVJFXImageDisplay;


/**
 * A class dedicated to basic JavaCV image operations.
 * @author Julien Seinturier - Universit&eacute; de Toulon / LIS umr CNRS 7020 - <a href="http://web.seinturier.fr">http://web.seinturier.fr</a>
 */
public class ImageBasic {
	
	/**
	 * The main method.
	 * @param args the main method arguments
	 */
    public static void main(String[] args) {
        String file = "data/images/lake-mountain.png";

        // Load an image, using its color model
        Mat image = imread(file, IMREAD_UNCHANGED);

        // The image load has failed, program exit
        if (image.empty()) System.exit(1);

        JavaCVJFXImageDisplay.display(file, image);

//        ex6();
//        ex7();
//        ex8();
//        ex9();
//        ex10();
//        ex11();
//        ex12();
//        ex13();
        ex14();
    }

}
