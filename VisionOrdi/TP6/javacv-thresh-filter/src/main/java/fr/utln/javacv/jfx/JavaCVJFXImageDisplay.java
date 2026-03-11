package fr.utln.javacv.jfx;

import org.bytedeco.javacpp.indexer.UByteRawIndexer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.jorigin.Common;
import org.jorigin.jfx.JImageCanvas;
import org.jorigin.jfx.JImageFeature;
import org.jorigin.jfx.JImageFeatureLayer;

import javafx.application.Application;
import javafx.application.Platform;
import javafx.geometry.Point2D;
import javafx.geometry.Rectangle2D;
import javafx.scene.Scene;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Label;
import javafx.scene.control.Separator;
import javafx.scene.image.Image;
import javafx.scene.image.PixelWriter;
import javafx.scene.image.WritableImage;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.paint.Color;
import javafx.scene.paint.Paint;
import javafx.scene.shape.Shape;
import javafx.scene.transform.Affine;
import javafx.stage.Screen;
import javafx.stage.Stage;

/**
 * An image display.
 * @author Julien Seinturier - Universit&eacute; de Toulon / LIS umr CNRS 7020 - <a href="http://web.seinturier.fr">http://web.seinturier.fr</a>
 *
 */
public class JavaCVJFXImageDisplay extends Application{

	private static JavaCVJFXImageDisplay me = null;
	
	private static boolean running = false;
	
	/**
	 * The image canvas.
	 */
	private JImageCanvas jfxCanvas = null;
	
	/**
	 * The primary stage
	 */
	private Stage jfxStage = null;
	
	/**
	 * The image to display
	 */
	private Image jfxImage = null;
	
	/**
	 * The title of the JavaFX frame.
	 */
	private String jfxFrameTitle = null;
	
	/**
	 * The feature layer that enables to display corners.
	 */
	private JImageFeatureLayer jfxFeatureLayer = null;
	
	/**
	 * Start the display by starting underlying JavaFX application.
	 */
	private static void startDisplay() {
		if (! running) {
			Common.init();

			Thread thread = new Thread() {
				public void run(){
					Application.launch(JavaCVJFXImageDisplay.class);
				}
			}; 
			
			thread.start();

			while((me == null) && (!running)){
				try {
					Thread.sleep(10);
				} catch (InterruptedException e) {
					System.err.println(e.getMessage());
					e.printStackTrace(System.err);
				}
			}
		}
	}
	
	/**
	 * Terminate the display by shutting down underlying JavaFX application.
	 */
	public static void stopDisplay() {

		if (running) {
			try {
				Platform.exit();
				running = false;
			} catch (Exception e) {
				System.err.println(e.getMessage());
				e.printStackTrace(System.err);
			}
		}

	}
	
	/**
	 * Default constructor, needed by JavaFX.
	 */
	public JavaCVJFXImageDisplay() {
	}
	
	@Override
	public void start(Stage primaryStage) throws Exception {
		
		// Take a screenshot
		Rectangle2D screenBounds = Screen.getPrimary().getBounds();
		
		// Create display
		this.jfxCanvas = new JImageCanvas();

		this.jfxCanvas.setBackgroundPaint(Color.DARKGRAY);
		
		this.jfxCanvas.setAutoFit(true);
		
		this.jfxFeatureLayer = new JImageFeatureLayer("Corners");
		
		this.jfxCanvas.addImageFeatureLayer(this.jfxFeatureLayer);
		
		BorderPane centerPane = new BorderPane();
		centerPane.setMinSize(0.0d, 0.0d);
		
		HBox bottomPane = new HBox();
		
		Label cursorPositionLB = new Label("Cursor position: -");
		Label imageSizeLB = new Label("Image size: - x - px");
        
		bottomPane.getChildren().add(imageSizeLB);
		bottomPane.getChildren().add(new Separator());
		bottomPane.getChildren().add(cursorPositionLB);
	
		centerPane.setCenter(this.jfxCanvas);
		centerPane.setBottom(bottomPane);
		
		Scene scene = new Scene(centerPane);

		primaryStage.setWidth(screenBounds.getWidth() / 2.0d);
		primaryStage.setHeight(screenBounds.getHeight() / 2.0d);
		primaryStage.setTitle("ImageDisplay");

		primaryStage.setScene(scene);
		
		this.jfxStage = primaryStage;
		
		JavaCVJFXImageDisplay.me = this;
		JavaCVJFXImageDisplay.running = true;	
	}
	
	/**
	 * Display the given {@link Mat OpenCV/JavaCV image} within the canvas.
	 * @param title the title of the frame
	 * @param image the image to display
	 */
	public static void display(String title, Mat image) {
		
		if (! running) {
			startDisplay();
		}
		
		JavaCVJFXImageDisplay jfxDisplay = JavaCVJFXImageDisplay.me;
		
		jfxDisplay.jfxImage = matToJavaFXImage(image);
		
		jfxDisplay.jfxFrameTitle = title;
		
		//jfxDisplay.jfxFeatureLayer.setStateDisplaying(false);
		jfxDisplay.jfxFeatureLayer.setImageFeatures(null);	
		
		Platform.runLater(() -> {jfxDisplay.updateDisplay();});
	}
	
	/**
	 * Update the display internally.
	 */
	private void updateDisplay() {
		 jfxStage.setTitle(jfxFrameTitle); 
		 jfxCanvas.setImage(jfxImage); 
		 jfxCanvas.refresh();
		 jfxStage.show();
	}
	
	/**
	 * Convert an {@link Mat OpenCV/JavaCV image} to an {@link Image JavaFX image}.
	 * @param mat the input OpenCV/JavaCV Mat that represents an image
	 * @return a JavaFX Image
	 * @throws IllegalArgumentException of input Mat is invalid (null or with an invalid size)
	 */
	private static Image matToJavaFXImage(Mat mat) {

		if (mat == null) {
			throw new IllegalArgumentException("Invalid null input Mat.");
		}
		
        int width = mat.cols();
        int height = mat.rows();

        if ((width < 1) || (height < 1)){
			throw new IllegalArgumentException("Invalid input Mat size "+width+"x"+height+".");
		}
        
        // Create a JavaFX WritableImage with the same dimensions as the Mat
        WritableImage writableImage = new WritableImage(width, height);
        
        // Get a JavaFX pixel writer that enables to modify image pixels
        PixelWriter pixelWriter = writableImage.getPixelWriter();
        
        // Gray image
        if (mat.channels() == 1) {
        	// Get the pixel data from the Mat
            byte[] data = new byte[width * height]; // 3 channels for RGB
            mat.data().get(data);

            // Write pixel values to the WritableImage
            int index = 0;
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int r = data[index] & 0xFF;
                    int g = r;
                    int b = r;
                    int argb = 0xFF000000 | (r << 16) | (g << 8) | b;
                    pixelWriter.setArgb(x, y, argb);
                    index++;
                }
            }
            
        // BGR Image
        } else if (mat.channels() == 3) {

        	// Access to the native image raster (the Mat)
            UByteRawIndexer indexer = mat.createIndexer();

            // Write pixel values to the WritableImage
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {

                    int argb = 0xFF000000 | (indexer.get(y, x, 2) << 16) | (indexer.get(y, x, 1) << 8) | indexer.get(y, x, 0);
                    pixelWriter.setArgb(x, y, argb);
                }
            }

        } else {
        	throw new IllegalArgumentException("Invalid input Mat type, only GRAY or RGB are supported.");
        }
        
        
        return writableImage;
    }
}

class CrossImageFeature implements JImageFeature {

	private JImageFeatureLayer layer = null;
	
	private Point2D center = null;
	
	private Color color = Color.CYAN;;
	
	private boolean displayable = true;
	
	private boolean displaying = true;
	
	/**
	 * Create a new cross centered on given coordinates
	 * @param x the X coordinate of the center
	 * @param y the Y coordinate of the center
	 */
	public CrossImageFeature(double x, double y) {
		center = new Point2D(x, y);
	}
	
	@Override
	public boolean isStateSelected() { return false; }

	@Override
	public void setStateSelected(boolean selected) {}

	@Override
	public boolean isStateSelectable() { return false; }

	@Override
	public void setStateSelectable(boolean selectable) {}

	@Override
	public boolean isStateDisplaying() { return displaying; }

	@Override
	public void setStateDisplaying(boolean displaying) {this.displaying = displaying;}

	@Override
	public boolean isStateDisplayable() {return displayable;}

	@Override
	public void setStateDisplayable(boolean displayable) {this.displayable = displayable;}

	@Override
	public void draw(GraphicsContext g2d, Affine transform) {
				
		if (g2d != null) {
			if (transform != null) {
				
				Affine orifginalTransform = g2d.getTransform();
				Paint originalFill = g2d.getFill();
				Paint originalStroke = g2d.getStroke();
									
				g2d.setTransform(new Affine());
				
				g2d.setStroke(color);
				
				Point2D pt = transform.transform(center);
				
				g2d.strokeLine(pt.getX()-4, pt.getY(), pt.getX()+4, pt.getY());
				g2d.strokeLine(pt.getX(), pt.getY()-4, pt.getX(), pt.getY()+4);
				
				g2d.setFill(originalFill);
				g2d.setStroke(originalStroke);
				g2d.setTransform(orifginalTransform);
			}
		}
	}

	@Override
	public boolean contains(double x, double y) { return false; }

	@Override
	public boolean contains(Shape s) { return false; }

	@Override
	public boolean intersects(Shape s) { return false; }

	@Override
	public boolean inside(Shape s) { return false; }

	@Override
	public Object getUserData() { return null; }

	@Override
	public void setUserData(Object data) {}

	@Override
	public JImageFeatureLayer getImageFeatureLayer() {
		return layer;
	}

	@Override
	public void setImageFeatureLayer(JImageFeatureLayer layer) {
		this.layer = layer;
	}
	
}

