package fr.utln.javacv.basic;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.javacpp.indexer.ByteIndexer;
import org.bytedeco.javacpp.indexer.DoubleIndexer;
import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.javacpp.indexer.UByteIndexer;

import static org.bytedeco.opencv.global.opencv_core.CV_8U;
import static org.bytedeco.opencv.global.opencv_core.CV_8S;
import static org.bytedeco.opencv.global.opencv_core.CV_16S;
import static org.bytedeco.opencv.global.opencv_core.CV_16U;
import static org.bytedeco.opencv.global.opencv_core.CV_32F;
import static org.bytedeco.opencv.global.opencv_core.CV_32S;
import static org.bytedeco.opencv.global.opencv_core.CV_64F;


/**
 * A utility class for {@link Mat}
 * @author Julien Seinturier - Universit&eacute; de Toulon / LIS umr CNRS 7020 - <a href="http://web.seinturier.fr">http://web.seinturier.fr</a>
 */
public class MatUtils {

	/**
	 * Create a new mat populated with random numbers.
	 * @param rows the number of rows
	 * @param cols the number of columns
	 * @param type the type
	 * @return a new Mat
	 */
	public static Mat rowMajorMat(int rows, int cols, int type) {
		Mat mat;
		
		mat = new Mat(rows, cols, type);
		
		int depth = mat.depth();
		
		
		if (depth == CV_8U) {
			UByteIndexer indexer = mat.createIndexer();
			int i = 1;
			for(int row = 0; row < mat.rows(); row++) {
				for(int col = 0; col < mat.cols(); col++) {
					for(int ch = 0; ch < mat.channels(); ch++) {
						indexer.put(row, col, ch, i);
						i++;
					}
				}
			}
		} else if (depth == CV_8S) {
			ByteIndexer indexer = mat.createIndexer();
			byte i = 1;
			for(int row = 0; row < mat.rows(); row++) {
				for(int col = 0; col < mat.cols(); col++) {
					for(int ch = 0; ch < mat.channels(); ch++) {
						indexer.put(row, col, ch, i);
						i++;
					}
				}
			}
		} else if (depth == CV_32F) {
			FloatIndexer indexer = mat.createIndexer();
			float i = 1;
			for(int row = 0; row < mat.rows(); row++) {
				for(int col = 0; col < mat.cols(); col++) {
					for(int ch = 0; ch < mat.channels(); ch++) {
						indexer.put(row, col, ch, i+((ch+1.0f)/10.0f));
					}
					i++;
				}
			}
		}  else if (depth == CV_64F) {
			DoubleIndexer indexer = mat.createIndexer();
			double i = 1;
			for(int row = 0; row < mat.rows(); row++) {
				for(int col = 0; col < mat.cols(); col++) {
					for(int ch = 0; ch < mat.channels(); ch++) {
						indexer.put(row, col, ch, i+((ch+1.0d)/10.0d));
					}
					i++;
				}
			}
		} else {
			throw new IllegalArgumentException("Invalid Mat type "+type);
		}
		
		return mat;
	}
}
