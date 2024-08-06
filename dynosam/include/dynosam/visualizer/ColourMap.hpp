/*
 *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Jesse Morris (jesse.morris@sydney.edu.au)
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a copy
 *   of this software and associated documentation files (the "Software"), to deal
 *   in the Software without restriction, including without limitation the rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:

 *   The above copyright notice and this permission notice shall be included in all
 *   copies or substantial portions of the Software.

 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *   SOFTWARE.
 */

#pragma once

#include "dynosam/common/Types.hpp"
#include "dynosam/utils/Numerical.hpp"

#include <opencv4/opencv2/opencv.hpp>
#include <opencv2/viz/types.hpp>
#include <glog/logging.h>

#include <cmath>

namespace dyno {

//Encapsulated differences between processing float/double depths and uint8_t
template<typename T, typename Enable = void>
struct color_traits;

template<typename T>
struct _floating_point_colour_traits {
  static constexpr inline T max() { return 1.0; }
};

template<typename T>
struct _int_point_colour_traits {
  static constexpr inline T max() { return 255; }
};

//Specalisation for integer types
template <typename T>
struct color_traits<T, std::enable_if_t<std::is_integral<T>::value>> : _int_point_colour_traits<T> {};
//Specalisation for floating point types
template <typename T>
struct color_traits<T, std::enable_if_t<std::is_floating_point<T>::value>> : _floating_point_colour_traits<T> {};

template<typename T>
struct RGBA {
  // Data.
  T r = 0;
  T g = 0;
  T b = 0;
  T a = max();

  RGBA() = default;
  RGBA(T r, T g, T b, T a = max()) : r(r), g(g), b(b), a(a) {}

  template<typename U>
  RGBA(const RGBA<U>& other) {
    //check if we need to go up (1 -> 255) or down (255 -> 1)
    constexpr auto other_max = RGBA<U>::max();

    //this is lower than other so we need to normalize using other max
    if constexpr (this->max() < other_max) {
      this->r = is_zero(other.r) ? 0 : static_cast<T>(other.r / other_max);
      this->g = is_zero(other.g) ? 0 : static_cast<T>(other.g / other_max);
      this->b = is_zero(other.b) ? 0 : static_cast<T>(other.b / other_max);
      this->a = is_zero(other.a) ? 0 : static_cast<T>(other.a / other_max);
      CHECK_LE(this->a, max());
    }
    //this is higher than other so we need to normalize using this max
    else {
      constexpr U cast_max = static_cast<U>(max());
      this->r = is_zero(other.r) ? 0 : static_cast<T>(other.r * cast_max);
      this->g = is_zero(other.g) ? 0 : static_cast<T>(other.g * cast_max);
      this->b = is_zero(other.b) ? 0 : static_cast<T>(other.b * cast_max);
      this->a = is_zero(other.a) ? 0 : static_cast<T>(other.a * cast_max);
      CHECK_LE(this->a, max());
    }
  }

  //! Maximum value (upper range) for the RGBA values given the type
  //! Should be 255u for integer types and 1.0f for floating point types
  static constexpr inline T max() { return color_traits<T>::max(); }
};


//From https://github.com/MIT-SPARK/Spark-DSG/blob/main/include/spark_dsg/color.h
struct Color : RGBA<uint8_t> {
  using RGBA<uint8_t>::max;

  // Constructors.
  Color() = default;
  Color(uint8_t r, uint8_t g, uint8_t b, uint8_t a = max()) : RGBA(r, g, b, a) {}
  virtual ~Color() = default;

  // Operators.
  bool operator==(const Color& other) const;
  bool operator!=(const Color& other) const { return !(*this == other); }
  bool operator<(const Color& other) const;
  bool operator>(const Color& other) const { return other < *this; }
  friend std::ostream& operator<<(std::ostream&, const Color&);

  // Operator to cv::Scalar in RGBA format
  operator cv::Scalar() const {
    return rgba();
  }

  /**
   * @brief Return in rgba format
   *
   * @return cv::Scalar
   */
  inline cv::Scalar rgba() const {
    return cv::Scalar(r, g, b, a);
  }

  /**
   * @brief Return in BGRA format.
   *
   * @return cv::Scalar
   */
  inline cv::Scalar bgra() const {
    return cv::Scalar(b, g, r, a);
  }


  /**
   * @brief Return the RGBA values as a double. Note this will change the range from 0-255 to 0-1.
   *
   * @return RGBA<double>
   */
  inline RGBA<double> toDouble() const {
    return RGBA<double>(*this);
  }

  // Tools.
  /**
   * @brief Blend another color with this one.
   * @param other The other color to blend.
   * @param weight The weight of the other color [0,1].
   * @return The resulting blended color.
   */
  Color blend(const Color& other, float weight = 0.5f) const;

  /**
   * @brief Merge another color into this one.
   * @param other The other color to merge.
   * @param weight The weight of the other color [0,1].
   */
  void merge(const Color& other, float weight = 0.5f);

  /**
   * @brief Hash function for colors to use colors as keys in maps, sets, etc.
   */
  struct Hash {
    uint32_t operator()(const Color& color) const {
      return color.r + (color.g << 8) + (color.b << 16) + (color.a << 24);
    }
  };

  // Presets.
  static Color black() { return Color(0, 0, 0); }
  static Color white() { return Color(255, 255, 255); }
  static Color red() { return Color(255, 0, 0); }
  static Color green() { return Color(0, 255, 0); }
  static Color blue() { return Color(0, 0, 255); }
  static Color yellow() { return Color(255, 255, 0); }
  static Color orange() { return Color(255, 127, 0); }
  static Color purple() { return Color(127, 0, 255); }
  static Color cyan() { return Color(0, 255, 255); }
  static Color magenta() { return Color(255, 0, 255); }
  static Color pink() { return Color(255, 127, 127); }
  static Color random();

  // Conversions.
  static Color fromHSV(float hue, float saturation, float value);
  static Color fromHLS(float hue, float luminance, float saturation);

  // Color maps.
  /**
   * @brief Generate a gray-scale color.
   * @param value The value of the gray [0,1, where 0 is black and 1 is white].
   */
  static Color gray(float value = 0.5f);

  /**
   * @brief Generate a color based on a quality value.
   * @param value The quality value [0,1], where 0 is red, 0.5 is yellow, and 1 is
   * green.
   */
  static Color quality(float value);

  /**
   * @brief Generate a color based on a spectrum of colors.
   * @param value The spectrum value [0,1], where 0 is the first color and 1 is the
   * last color.
   * @param colors The list of colors in the spectrum. Colors are assumed equidistant.
   */
  static Color spectrum(float value, const std::vector<Color>& colors);

  /**
   * @brief Generate a color based on a ironbow value.
   * @param value The temperature value [0,1], where 0 is dark and 1 is light.
   */
  static Color ironbow(float value);

  /**
   * @brief Generate a color based on a rainbow value.
   * @param value The rainbow value [0,1], where 0 is red, 0.5 is green, and 1 is
   * blue.
   */
  static Color rainbow(float value);

  /**
   * @brief Generate a sequence of never repeating colors in the rainbow spectrum.
   * @param id The id of the color in the sequence.
   * @param ids_per_revolution The number of colors per revolution of the hue.
   */
  static Color rainbowId(size_t id, size_t ids_per_revolution = 16);

  /**
   * @brief Generates a sequence of never repeating colors using the golden ration to seperate
   * the colours along the HSV ring spectrum.
   *
   * @param id The id of the color in the sequence.
   * @param saturation Saturation value
   * @param value Value value ;)
   * @return Color
   */
  static Color uniqueId(size_t id, float saturation = 0.5f, float value = 0.95f);

 private:
  static const std::vector<Color> ironbow_colors_;
};


// struct ColourMap {

// constexpr static int COLOURS_MAX = 150;

// #define GET_HUE(hsv) hsv(0)
// #define GET_SATURATION(hsv) hsv(1)
// #define GET_VALUE(hsv) hsv(2)

// #define GET_R(rgb) rgb(0)
// #define GET_G(rgb) rgb(1)
// #define GET_B(rgb) rgb(2)


// /**
//  * @brief Converts HSV to RGB
//  *
//  * H is an angle in degrees
//  * S, V can either be between 0-1 or 0-255
//  *
//  * https://stackoverflow.com/questions/3018313/algorithm-to-convert-rgb-to-hsv-and-hsv-to-rgb-in-range-0-255-for-both
//  *
//  * @param hsv
//  * @param use_opencv_convention
//  * @return cv::Scalar
//  */
// static cv::Scalar HSV2RGB(cv::Scalar hsv, bool use_opencv_convention = false);

// static inline cv::Scalar getRainbowColour(size_t index) {
//     CHECK(index < COLOURS_MAX);
//     return cv::Scalar(
//         colmap[index][0] * 255.0,
//         colmap[index][1] * 255.0,
//         colmap[index][2] * 255.0,
//         255.0);
// }

// static inline cv::Scalar generateUniqueColour(
//     int id,
//     bool use_opencv_convention = false,
//     double saturation = 0.5,
//     double value = 0.95) {
//   constexpr static double phi = (1 + std::sqrt(5.0))/2.0;
//   auto n = (double)id * phi - std::floor((double)id * phi);

//   //want in 255 mode
//   double hue = std::floor(n * 256.0);
//   return HSV2RGB(cv::Scalar(hue, saturation, value), use_opencv_convention);
// }

// //if use_opencv_convention then colours are reversed to be BGRA
// //expects colour to be RGBA
// static inline cv::Scalar RGBA2BGRA(cv::Scalar colour, bool use_opencv_convention = false) {
//   if(use_opencv_convention) {
//     return cv::Scalar(colour(2), colour(1), colour(0), colour(3));
//   }
//   return colour;
// }


// static inline cv::Scalar getObjectColour(ObjectId label, bool use_opencv_convention = false) {
//   return generateUniqueColour((int)label, use_opencv_convention);
//   // while (label > 25)
//   // {
//   //   label = label / 2.0;
//   // }

//   // cv::Scalar colour;
//   // switch (label)
//   // {
//   //   case 0:
//   //     colour = cv::Scalar(0, 0, 255, 255);  // red
//   //     break;
//   //   case 1:
//   //     colour = cv::Scalar(128, 0, 128, 255);  // 255, 165, 0
//   //     break;
//   //   case 2:
//   //     colour =  cv::Scalar(0, 255, 255, 255);
//   //     break;
//   //   case 3:
//   //     colour =  cv::Scalar(0, 255, 0, 255);  // 255,255,0
//   //     break;
//   //   case 4:
//   //     colour =  cv::Scalar(255, 0, 0, 255);  // 255,192,203
//   //     break;
//   //   case 5:
//   //     colour =  cv::Scalar(0, 255, 255, 255);
//   //     break;
//   //   case 6:
//   //     colour =  cv::Scalar(128, 0, 128, 255);
//   //     break;
//   //   case 7:
//   //     colour =  cv::Scalar(255, 255, 255, 255);
//   //      break;
//   //   case 8:
//   //     colour =  cv::Scalar(255, 228, 196, 255);
//   //     break;
//   //   case 9:
//   //     colour =  cv::Scalar(180, 105, 255, 255);
//   //      break;
//   //   case 10:
//   //     colour =  cv::Scalar(165, 42, 42, 255);
//   //      break;
//   //   case 11:
//   //     colour =  cv::Scalar(35, 142, 107, 255);
//   //      break;
//   //   case 12:
//   //     colour =  cv::Scalar(45, 82, 160, 255);
//   //      break;
//   //   case 13:
//   //     colour =  cv::Scalar(0, 0, 255, 255);  // red
//   //      break;
//   //   case 14:
//   //     colour =  cv::Scalar(255, 165, 0, 255);
//   //      break;
//   //   case 15:
//   //     colour =  cv::Scalar(0, 255, 0, 255);
//   //      break;
//   //   case 16:
//   //     colour =  cv::Scalar(255, 255, 0, 255);
//   //      break;
//   //   case 17:
//   //     colour =  cv::Scalar(255, 192, 203, 255);
//   //      break;
//   //   case 18:
//   //     colour =  cv::Scalar(10, 255, 0, 255);
//   //      break;
//   //   case 19:
//   //     colour =  cv::Scalar(128, 0, 128, 255);
//   //      break;
//   //   case 20:
//   //     colour =  cv::Scalar(125, 255, 75, 255);
//   //      break;
//   //   case 21:
//   //     colour =  cv::Scalar(255, 228, 196, 255);
//   //      break;
//   //   case 22:
//   //     colour =  cv::Scalar(180, 105, 255, 255);
//   //      break;
//   //   case 23:
//   //     colour =  cv::Scalar(165, 42, 42, 255);
//   //      break;
//   //   case 24:
//   //     colour =  cv::Scalar(35, 142, 107, 255);
//   //      break;
//   //   case 25:
//   //     colour =  cv::Scalar(45, 82, 160, 255);
//   //      break;
//   //   case 41:
//   //     colour =  cv::Scalar(60, 20, 220, 255);
//   //      break;
//   //   default:
//   //     LOG(WARNING) << "Label is " << label;
//   //     colour =  cv::Scalar(60, 20, 220, 255);
//   //      break;
//   // }
// }

// static constexpr double colmap[COLOURS_MAX][3] = { { 0, 0, 0.5263157895 },
//                                  { 0, 0, 0.5526315789 },
//                                  { 0, 0, 0.5789473684 },
//                                  { 0, 0, 0.6052631579 },
//                                  { 0, 0, 0.6315789474 },
//                                  { 0, 0, 0.6578947368 },
//                                  { 0, 0, 0.6842105263 },
//                                  { 0, 0, 0.7105263158 },
//                                  { 0, 0, 0.7368421053 },
//                                  { 0, 0, 0.7631578947 },
//                                  { 0, 0, 0.7894736842 },
//                                  { 0, 0, 0.8157894737 },
//                                  { 0, 0, 0.8421052632 },
//                                  { 0, 0, 0.8684210526 },
//                                  { 0, 0, 0.8947368421 },
//                                  { 0, 0, 0.9210526316 },
//                                  { 0, 0, 0.9473684211 },
//                                  { 0, 0, 0.9736842105 },
//                                  { 0, 0, 1 },
//                                  { 0, 0.0263157895, 1 },
//                                  { 0, 0.0526315789, 1 },
//                                  { 0, 0.0789473684, 1 },
//                                  { 0, 0.1052631579, 1 },
//                                  { 0, 0.1315789474, 1 },
//                                  { 0, 0.1578947368, 1 },
//                                  { 0, 0.1842105263, 1 },
//                                  { 0, 0.2105263158, 1 },
//                                  { 0, 0.2368421053, 1 },
//                                  { 0, 0.2631578947, 1 },
//                                  { 0, 0.2894736842, 1 },
//                                  { 0, 0.3157894737, 1 },
//                                  { 0, 0.3421052632, 1 },
//                                  { 0, 0.3684210526, 1 },
//                                  { 0, 0.3947368421, 1 },
//                                  { 0, 0.4210526316, 1 },
//                                  { 0, 0.4473684211, 1 },
//                                  { 0, 0.4736842105, 1 },
//                                  { 0, 0.5, 1 },
//                                  { 0, 0.5263157895, 1 },
//                                  { 0, 0.5526315789, 1 },
//                                  { 0, 0.5789473684, 1 },
//                                  { 0, 0.6052631579, 1 },
//                                  { 0, 0.6315789474, 1 },
//                                  { 0, 0.6578947368, 1 },
//                                  { 0, 0.6842105263, 1 },
//                                  { 0, 0.7105263158, 1 },
//                                  { 0, 0.7368421053, 1 },
//                                  { 0, 0.7631578947, 1 },
//                                  { 0, 0.7894736842, 1 },
//                                  { 0, 0.8157894737, 1 },
//                                  { 0, 0.8421052632, 1 },
//                                  { 0, 0.8684210526, 1 },
//                                  { 0, 0.8947368421, 1 },
//                                  { 0, 0.9210526316, 1 },
//                                  { 0, 0.9473684211, 1 },
//                                  { 0, 0.9736842105, 1 },
//                                  { 0, 1, 1 },
//                                  { 0.0263157895, 1, 0.9736842105 },
//                                  { 0.0526315789, 1, 0.9473684211 },
//                                  { 0.0789473684, 1, 0.9210526316 },
//                                  { 0.1052631579, 1, 0.8947368421 },
//                                  { 0.1315789474, 1, 0.8684210526 },
//                                  { 0.1578947368, 1, 0.8421052632 },
//                                  { 0.1842105263, 1, 0.8157894737 },
//                                  { 0.2105263158, 1, 0.7894736842 },
//                                  { 0.2368421053, 1, 0.7631578947 },
//                                  { 0.2631578947, 1, 0.7368421053 },
//                                  { 0.2894736842, 1, 0.7105263158 },
//                                  { 0.3157894737, 1, 0.6842105263 },
//                                  { 0.3421052632, 1, 0.6578947368 },
//                                  { 0.3684210526, 1, 0.6315789474 },
//                                  { 0.3947368421, 1, 0.6052631579 },
//                                  { 0.4210526316, 1, 0.5789473684 },
//                                  { 0.4473684211, 1, 0.5526315789 },
//                                  { 0.4736842105, 1, 0.5263157895 },
//                                  { 0.5, 1, 0.5 },
//                                  { 0.5263157895, 1, 0.4736842105 },
//                                  { 0.5526315789, 1, 0.4473684211 },
//                                  { 0.5789473684, 1, 0.4210526316 },
//                                  { 0.6052631579, 1, 0.3947368421 },
//                                  { 0.6315789474, 1, 0.3684210526 },
//                                  { 0.6578947368, 1, 0.3421052632 },
//                                  { 0.6842105263, 1, 0.3157894737 },
//                                  { 0.7105263158, 1, 0.2894736842 },
//                                  { 0.7368421053, 1, 0.2631578947 },
//                                  { 0.7631578947, 1, 0.2368421053 },
//                                  { 0.7894736842, 1, 0.2105263158 },
//                                  { 0.8157894737, 1, 0.1842105263 },
//                                  { 0.8421052632, 1, 0.1578947368 },
//                                  { 0.8684210526, 1, 0.1315789474 },
//                                  { 0.8947368421, 1, 0.1052631579 },
//                                  { 0.9210526316, 1, 0.0789473684 },
//                                  { 0.9473684211, 1, 0.0526315789 },
//                                  { 0.9736842105, 1, 0.0263157895 },
//                                  { 1, 1, 0 },
//                                  { 1, 0.9736842105, 0 },
//                                  { 1, 0.9473684211, 0 },
//                                  { 1, 0.9210526316, 0 },
//                                  { 1, 0.8947368421, 0 },
//                                  { 1, 0.8684210526, 0 },
//                                  { 1, 0.8421052632, 0 },
//                                  { 1, 0.8157894737, 0 },
//                                  { 1, 0.7894736842, 0 },
//                                  { 1, 0.7631578947, 0 },
//                                  { 1, 0.7368421053, 0 },
//                                  { 1, 0.7105263158, 0 },
//                                  { 1, 0.6842105263, 0 },
//                                  { 1, 0.6578947368, 0 },
//                                  { 1, 0.6315789474, 0 },
//                                  { 1, 0.6052631579, 0 },
//                                  { 1, 0.5789473684, 0 },
//                                  { 1, 0.5526315789, 0 },
//                                  { 1, 0.5263157895, 0 },
//                                  { 1, 0.5, 0 },
//                                  { 1, 0.4736842105, 0 },
//                                  { 1, 0.4473684211, 0 },
//                                  { 1, 0.4210526316, 0 },
//                                  { 1, 0.3947368421, 0 },
//                                  { 1, 0.3684210526, 0 },
//                                  { 1, 0.3421052632, 0 },
//                                  { 1, 0.3157894737, 0 },
//                                  { 1, 0.2894736842, 0 },
//                                  { 1, 0.2631578947, 0 },
//                                  { 1, 0.2368421053, 0 },
//                                  { 1, 0.2105263158, 0 },
//                                  { 1, 0.1842105263, 0 },
//                                  { 1, 0.1578947368, 0 },
//                                  { 1, 0.1315789474, 0 },
//                                  { 1, 0.1052631579, 0 },
//                                  { 1, 0.0789473684, 0 },
//                                  { 1, 0.0526315789, 0 },
//                                  { 1, 0.0263157895, 0 },
//                                  { 1, 0, 0 },
//                                  { 0.9736842105, 0, 0 },
//                                  { 0.9473684211, 0, 0 },
//                                  { 0.9210526316, 0, 0 },
//                                  { 0.8947368421, 0, 0 },
//                                  { 0.8684210526, 0, 0 },
//                                  { 0.8421052632, 0, 0 },
//                                  { 0.8157894737, 0, 0 },
//                                  { 0.7894736842, 0, 0 },
//                                  { 0.7631578947, 0, 0 },
//                                  { 0.7368421053, 0, 0 },
//                                  { 0.7105263158, 0, 0 },
//                                  { 0.6842105263, 0, 0 },
//                                  { 0.6578947368, 0, 0 },
//                                  { 0.6315789474, 0, 0 },
//                                  { 0.6052631579, 0, 0 },
//                                  { 0.5789473684, 0, 0 },
//                                  { 0.5526315789, 0, 0 } };

// };

} //dyno
