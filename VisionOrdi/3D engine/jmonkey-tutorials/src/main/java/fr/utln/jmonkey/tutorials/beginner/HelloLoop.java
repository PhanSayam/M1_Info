package fr.utln.jmonkey.tutorials.beginner;

import com.jme3.app.SimpleApplication;
import com.jme3.material.Material;
import com.jme3.math.ColorRGBA;
import com.jme3.math.FastMath;
import com.jme3.scene.Geometry;
import com.jme3.scene.shape.Box;

public class HelloLoop extends SimpleApplication {

    private Geometry cubeSlow;
    private Geometry cubeFast;
    private Geometry pulsateCube;
    private Geometry colorCube;
    private Geometry rollCube;

    private float time = 0;

    public static void main(String[] args){
        HelloLoop app = new HelloLoop();
        app.start();
    }

    public HelloLoop(){
    }

    @Override
    public void simpleInitApp() {
        cubeSlow = createCube(ColorRGBA.Blue, -2, 0, 0);
        cubeFast = createCube(ColorRGBA.Red, 2, 0, 0);
        pulsateCube = createCube(ColorRGBA.Green, 0, 2, 0);
        colorCube = createCube(ColorRGBA.White, 0, -2, 0);
        rollCube = createCube(ColorRGBA.Yellow, 0, 0, -2);

        rootNode.attachChild(cubeSlow);
        rootNode.attachChild(cubeFast);
        rootNode.attachChild(pulsateCube);
        rootNode.attachChild(colorCube);
        rootNode.attachChild(rollCube);
    }

    private Geometry createCube(ColorRGBA color, float x, float y, float z) {
        Box b = new Box(0.5f, 0.5f, 0.5f);
        Geometry geom = new Geometry("Cube", b);
        Material mat = new Material(assetManager, "Common/MatDefs/Misc/Unshaded.j3md");
        mat.setColor("Color", color);
        geom.setMaterial(mat);
        geom.setLocalTranslation(x, y, z);
        return geom;
    }

    @Override
    public void simpleUpdate(float tpf) {
        time += tpf;

        cubeSlow.rotate(0, -1.0f * tpf, 0);

        cubeFast.rotate(0, 2.0f * tpf, 0);

        float scale = 1.0f + FastMath.sin(time * FastMath.TWO_PI) * 0.5f;
        pulsateCube.setLocalScale(scale);

        float r = FastMath.abs(FastMath.sin(time));
        float g = FastMath.abs(FastMath.cos(time));
        float b = FastMath.abs(FastMath.sin(time * 0.5f));
        colorCube.getMaterial().setColor("Color", new ColorRGBA(r, g, b, 1.0f));

        rollCube.rotate(2.0f * tpf, 0, 0);
        rollCube.move(0, 0, 2.0f * tpf);
    }
}