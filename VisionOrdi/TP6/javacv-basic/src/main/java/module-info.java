
/**
 * The module description.
 * @author Julien Seinturier - Universit&eacute; de Toulon / LIS umr CNRS 7020 - <a href="http://web.seinturier.fr">http://web.seinturier.fr</a>
 */
module fr.utln.javacv {

	exports fr.utln.javacv.basic;
	exports fr.utln.javacv.jfx;

	requires transitive org.jcommon.jfx;
	
	requires transitive javafx.base;
	requires transitive javafx.controls;
	requires transitive javafx.graphics;
	
	requires transitive java.desktop;
	requires transitive java.logging;
	
	requires transitive org.bytedeco.javacv;
	requires transitive org.bytedeco.opencv;
	requires transitive org.bytedeco.javacpp;
}