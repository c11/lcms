package net.larse.lcms.helper;

/** Static array manipulation functions. */
public class ArrayHelper {
  /**
   * Count the number of occurrences of value in array, between start (incl) and end (excl). if end
   * is negative, it is taken as the number of entries from the end (ie: -1 = len-1).
   */
  public static int count(int value, int[] array, int start, int end) {
    if (end < 0) {
      end = array.length + end;
    }
    int count = 0;
    for (int i = start; i < end; i++) {
      if (array[i] == value) {
        count++;
      }
    }
    return count;
  }

  /**
   * Find the first occurrence of value in array, between start (incl) and end (excl). if end is
   * negative, it is taken as the number of entries from the end (ie: -1 = len-1).
   */
  public static int first(int value, int[] array, int start, int end) {
    if (end < 0) {
      end = array.length + end;
    }
    for (int i = start; i < end; i++) {
      if (array[i] == value) {
        return i;
      }
    }
    return -1;
  }

  /**
   * Find the last occurrence of value in array, between start (incl) and end (excl). if end is
   * negative, it is taken as the number of entries from the end (ie: -1 = len-1).
   */
  public static int last(int value, int[] array, int start, int end) {
    if (end < 0) {
      end = array.length + end;
    }
    for (int i = end - 1; i >= start; i--) {
      if (array[i] == value) {
        return i;
      }
    }
    return -1;
  }
}
