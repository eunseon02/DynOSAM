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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <exception>

#include <optional>
#include <atomic>

//forward
struct ExcpetionHook;

// class ExceptionStream {


// public:
//     template<typename Exception = std::runtime_error>
//     ExceptionStream() {
//         throw_exception_ = [=]() {
//             throw Exception(ss_.str());
//         };
//     }

//     ~ExceptionStream() {
//         LOG(INFO) << "Destructor";
//         throw_exception_();
//     }

//     ExceptionStream(const ExceptionStream& other) {
//         //copy
//         LOG(INFO) << "copy";
//         ss_ << other.ss_.str();
//         throw_exception_ = other.throw_exception_;
//     }

//     ExceptionStream& operator=(const ExceptionStream& other) {
//         LOG(INFO) << "Assignment";
//         //we want copy and swap idiom
//         ss_.clear();
//         ExceptionStream tmp_other(other);
//         tmp_other.swap(*this);
//         return *this;
//     }

//     template<typename T>
//     ExceptionStream& operator<<(const T& t){
//         ss_ << t;
//         return *this;
//     }


//     void swap(ExceptionStream& other) {
//         using std::swap;
//         swap(ss_, other.ss_);
//         swap(throw_exception_, other.throw_exception_);
//     }

// private:

// private:
//     std::stringstream ss_;
//     std::function<void()> throw_exception_;

// };

// class D {
// public:
//     D() { LOG(INFO) << "Makign D"; }
//     ~D() { LOG(INFO) << ss.str(); }

//     template<typename T>
//     D& operator<<(const T& t){
//         ss << t;
//         return *this;
//     }

//     D& operator=(const D& d) {
//         //we want copy and swap idiom
//         LOG(INFO) << "assignment";
//         ss.clear();
//         D(d).swap(*this);
//         // //first clear
//         // ss.clear();
//         // //copy
//         // //swap
//         // ss.str(d.ss.str());
//         return *this;
//     }

//     D(D&& d) {
//         LOG(INFO) << "move";
//         D(d).swap(*this);
//     }

//     D& operator=(D&& d) {
//         LOG(INFO) << "move assignment";
//         D(std::move(d)).swap(*this);
//         return *this;
//     }

//     D(const D& d) {
//         //copy
//         LOG(INFO) << "copy";
//         ss << d.ss.str();
//     }

//     void swap(D& d) {
//         using std::swap;
//         swap(ss, d.ss);
//     }

//     std::stringstream ss;
// };

// D throwOnCondition(bool condition) {
//     if(condition) {

//     }
// }

// struct ExcpetionHook {

//     // template<typename T>
//     // Exceptions& operator<<(const T& t){
//     //     return this->add(t);
//     // }

//     ExceptionStream make() {
//         LOG(INFO) << "here";
//         ExceptionStream d;
//         d << "yes hello " << 10;
//         LOG(INFO) << "now here";
//         return d;
//     }

//     // D make() {
//     //     D d;
//     //     d << "In make ";
//     //     return d;
//     // }

// };


// TEST(CodeConcepts, exceptionsWithStream) {
//     ExcpetionHook e;
//     e.make() << " bad error";
//     LOG(INFO) << "tmp obj";
//     // D d1(d);
// }

TEST(CodeConcepts, testModifyOptionalString) {

    auto modify = [](std::optional<std::reference_wrapper<std::string>> string) {
        if(string) {
           string->get() = "udpated";
        }
    };

    std::string input = "before";
    modify(input);

    EXPECT_EQ(input, "udpated");
}
