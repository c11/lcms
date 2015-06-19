/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package net.larse.lcms.algorithms;

import net.larse.lcms.helper.Raster;
import org.junit.*;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.*;

import static org.junit.Assert.*;

public class VCTTest {
  String mask_file;
  String b3_file;
  String b4_file;
  String b5_file;
  String b6_file;
  String b7_file;
  String dnbr_file;
  String ndvi_file;

  String expected_mask_file;

  int[] years;

  public VCTTest() {
  }
  
  @BeforeClass
  public static void setUpClass() {
    setRemap();
  }
  
  @AfterClass
  public static void tearDownClass() {
  }

  @Before
  public void setUp() throws Exception {
    mask_file = "./lcms/test/test_vct/inputs/input_tsa_45030_mask.tif";
    b3_file = "./lcms/test/test_vct/inputs/input_tsa_45030_udist_b3.tif";
    b4_file = "./lcms/test/test_vct/inputs/input_tsa_45030_udist_b4.tif";
    b5_file = "./lcms/test/test_vct/inputs/input_tsa_45030_udist_b5.tif";
    b6_file = "./lcms/test/test_vct/inputs/input_tsa_45030_udist_b6.tif";
    b7_file = "./lcms/test/test_vct/inputs/input_tsa_45030_udist_b7.tif";
    dnbr_file = "./lcms/test/test_vct/inputs/input_tsa_45030_udist_dnbr.tif";
    ndvi_file = "./lcms/test/test_vct/inputs/input_tsa_45030_udist_ndvi.tif";

    expected_mask_file = "./lcms/test/test_vct/outputs/output_tsa_45030_dist_mask.tif";

    years = new int[] {2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012};
  }

  @After
  public void tearDown() {
  }

  public class Record {
    public int year;
    public String variable;
    public String fn;
    public Raster raster;

    public Record(Iterable<String> tokens) {
      Iterator<String> it = tokens.iterator();
      this.year = Integer.parseInt(it.next());
      this.variable = it.next();
      this.fn = it.next();
      this.raster = new Raster(this.fn);
    }
  }

  private static final Map<Integer, Integer> MASK_MAP = new HashMap<>();

  public static void setRemap() {
    MASK_MAP.put(0, 0); // BACKGROUND -> BACKGROUND
    MASK_MAP.put(254, 1); // MASK_BAD_VALUE -> BACKGROUND
    MASK_MAP.put(255, 1); // MASK_MISSING_VALUE -> BACKGROUND
    MASK_MAP.put(5, 1); // CLOUD -> CLOUD
    MASK_MAP.put(4, 2); // CLOUD_EDGE -> CLOUD_EDGE
    MASK_MAP.put(2, 3); // SHADOW -> SHADOW
    MASK_MAP.put(3, 4); // SHADOW_EDGE -> SHADOW_EDGE
    MASK_MAP.put(7, 5); // SNOW -> SNOW
    MASK_MAP.put(1, 6); // WATER -> WATER
    MASK_MAP.put(8, 7); // CLEAR_LAND -> CLEAR_LAND
    MASK_MAP.put(9, 8); // CORE_FOREST -> CORE_FOREST
    MASK_MAP.put(10, 9); // CORE_NONFOREST -> CORE_NONFOREST
    MASK_MAP.put(11, 10); // CONFIDENT_CLEAR -> CONFIDENT_CLEAR
    MASK_MAP.put(21, 11); // CONFIDENT_NONCLOUD -> CONFIDENT_NONCLOUD
    MASK_MAP.put(22, 12); // CONFIDENT_NONSHADOW -> CONFIDENT_NONSHADOW
  }

  public void remapMask(int[] mask) {
    for (int i = 0; i < mask.length; i++) {
      mask[i] = MASK_MAP.get(mask[i]);
    }
  }
  
  public void scaleUdVariables(double[][] ud) {
    final double SCALE = 0.01;
    final double OFFSET = -100.5;
    final int N_BANDS = ud.length;
    final int N_YEARS = ud[0].length;
    final List<Integer> UD_INDEXES = Arrays.asList(0, 2, 3);
    
    double[] s = {SCALE, SCALE, SCALE, SCALE, SCALE, SCALE, SCALE};
    double[] t = {0.0, 0.0, 0.0, 0.0, 0.0, OFFSET, OFFSET};
    for (int i = 0; i < N_YEARS; i++) {
      for (int j = 0; j < N_BANDS - 1; j++) {
        double tmp = s[j] * (ud[j][i] + t[j]);
        ud[j][i] = tmp;
      }
    }
    
    // UD composite band
    for (int i = 0; i < N_YEARS; i++) {
      double sumSq = 0.0;
      for (Integer index : UD_INDEXES) {
        double tmp = ud[index][i];
        tmp = tmp >= 0.0 ? tmp : tmp / 2.5;
        sumSq += tmp * tmp;
      }
      ud[N_BANDS - 1][i] = Math.sqrt(sumSq / UD_INDEXES.size());
    }
  }
  
  public int[][] scaleResult(VCT.VCTOutput result) {
    int[][] output = new int[5][result.distFlag.length];
    output[0] = result.distFlag;
    for (int i=0; i<result.distMagn.length; i++) {
      if (result.distMagn[i] == -1.0) {
        output[1][i] = 0;
      } else {
        output[1][i] = (int) (result.distMagn[i] / 0.1);
      }
      if (result.distMagnVi[i] == -1.0) {
        output[2][i] = 0;
      } else {
        output[2][i] = (int) ((result.distMagnVi[i] / 0.01) + 100.5);
      }
      
      if (result.distMagnBr[i] == -1.0) {
        output[3][i] = 0;
      } else {
        output[3][i] = (int) ((result.distMagnBr[i] / 0.01) + 100.5);
      }
      
      if (result.distMagnB4[i] == -1.0) {
        output[4][i] = 0;
      } else {
        output[4][i] = (int) ((result.distMagnB4[i] / 0.1) + 100.5);
      }
    }
    return output;
  }
  
  @Test
  public void testSpatial() throws FileNotFoundException, IOException {
    Raster mask = new Raster(mask_file);
    Raster b3 = new Raster(b3_file);
    Raster b4 = new Raster(b4_file);
    Raster b5 = new Raster(b5_file);
    Raster b6 = new Raster(b6_file);
    Raster b7 = new Raster(b7_file);
    Raster dnbr = new Raster(dnbr_file);
    Raster ndvi = new Raster(ndvi_file);

    Raster expected_mask = new Raster(expected_mask_file);

    VCT vct = new VCT();


    double mapX = mask.ulx;
    double mapY = mask.uly;
    int xsize = mask.ds.getRasterXSize();
    int ysize = mask.ds.getRasterYSize();
    int nYears = mask.bands.length;

    //a necessary step to use the current mask.
    int[] intMask = new int[nYears];
    int[] intExpected = new int[nYears];

    //B3, B4, B5, B7, thermal, NDVI, DNBR, COMP
    double[][] ud = new double[8][nYears];

    int failedPixel = 0;
    for (int y = 0; y < ysize; y++) {
      for (int x = 0; x < xsize; x++) {
        double mapx = mapX + x * mask.pixelX;
        double mapy = mapY + y * mask.pixelY;

        double[] maskDat = mask.Read(mapx, mapy, 1,1);
        for (int i = 0; i < maskDat.length; i++) {
          intMask[i] = (int)maskDat[i];
        }
        remapMask(intMask);

        ud[0] = b3.Read(mapx, mapy, 1,  1);
        ud[1] = b4.Read(mapx, mapy, 1,  1);
        ud[2] = b5.Read(mapx, mapy, 1,  1);
        ud[3] = b6.Read(mapx, mapy, 1,  1);
        ud[4] = b7.Read(mapx, mapy, 1,  1);

        ud[5] = ndvi.Read(mapx, mapy, 1, 1);
        ud[6] = dnbr.Read(mapx, mapy, 1, 1);

        scaleUdVariables(ud);


        VCT.VCTOutput output = vct.getResult(ud, intMask, years);

        //In this test, we are only focusing on the disturbance label
        double[] expected = expected_mask.Read(mapx, mapy, 1, 1);
        for (int i = 0; i < expected.length; i++) {
          intExpected[i] = (int)expected[i];
        }
        if (!Arrays.equals(intExpected, output.distFlag)) {
          failedPixel++;
        }
      }
    }
    assertEquals(String.format("Total failed: %d", failedPixel), failedPixel, 0);
    System.out.println(String.format("Total failed: %d", failedPixel));
  }
}
