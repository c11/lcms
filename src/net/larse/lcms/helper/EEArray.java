package net.larse.lcms.helper;

/**
 * This is used to mimic com.google.earthengine.api.array.EEArray
 */

//import java.util.List;

public class EEArray {
  private double[] array;

  public EEArray(double[] array) {
    this.array = array;
  }

  public static class Builder {
    private double[] array;

    public Builder(PixelType type, int[] lengths) {
      int size = 1;
      for (int i = 0; i < lengths.length; i++) {
        size *= lengths[i];
      }
      this.array = new double[size];
    }

    public EEArray build() {
      return new EEArray(array);
    }

    public void setDouble(double value, int offset) {
      array[offset] = value;
    }
  }

  public static EEArray.Builder builder(PixelType type, int... lengths) {
    return new Builder(type, lengths);
  }
}

