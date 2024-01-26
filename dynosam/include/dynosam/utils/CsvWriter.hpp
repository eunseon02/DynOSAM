/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris (jesse.morris@sydney.edu.au)
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

#include "dynosam/common/Exceptions.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <typeinfo>
#include <vector>
#include <tuple>



namespace dyno {

struct InvalidCsvHeaderException : public DynosamException {
InvalidCsvHeaderException(const std::string& what) : DynosamException(what) {}
};



//cannot be constructed with num args < 1
class CsvHeader : public std::vector<std::string> {

public:

    template<typename...HeaderArgs, typename = std::enable_if_t<(std::is_convertible<HeaderArgs, std::string>::value && ...)>>
    CsvHeader(const HeaderArgs&... args) {
        std::vector<std::string>* result = this;
        std::apply([result](const auto&... elems) {
            result->reserve(sizeof...(elems));
            (result->push_back(std::string(elems)), ...);
        }, std::forward_as_tuple(args...));
    }

    std::string toString(const std::string& seperator = " ") const {
        std::stringstream ss;
        size_t i = 0;
        for(; i < size() - 1u; i++) {
            ss << this->at(i);
            ss << seperator;
        }
        ss << this->at(i);
        return ss.str();
    }



};


// Modified from: https://github.com/al-eax/CSVWriter/blob/master/include/CSVWriter.h
class CsvWriter
{
    public:

        CsvWriter(const CsvHeader& header, const std::string& seperator = ",");
        ~CsvWriter();


        CsvWriter& add(const char *str);

        CsvWriter& add(char *str);

        CsvWriter& add(const std::string& str);

        template<typename T>
        CsvWriter& add(const T& str){
                //if autoNewRow is enabled, check if we need a line break
            if(this->value_count_ == this->column_number_ ){
                this->newRow();
            }

            if(value_count_ > 0) this->ss_ << this->seperator_;
            this->ss_ << str;
            this->value_count_++;

            return *this;
        }

        template<typename T>
        CsvWriter& operator<<(const T& t){
            return this->add(t);
        }

        void operator+=(CsvWriter &csv){
            this->ss_ << std::endl << csv;
        }

        std::string toString() const {
            return ss_.str();
        }

        friend std::ostream& operator<<(std::ostream& os, const CsvWriter& csv){
            return os << csv.toString();
        }

        CsvWriter& newRow();

        bool write(const std::string& filename) const;
        bool write(const std::string& filename, bool append) const;
        bool write(std::ostream& stream) const;

        // void enableAutoNewRow(int numberOfColumns){
        //     this->column_number_ = numberOfColumns;
        // }

        // void disableAutoNewRow(){
        //     this->column_number_ = -1;
        // }
       //you can use this reset method in destructor if you gonna use it in heap mem.
        void resetContent();

    protected:
        const CsvHeader header_;
        const size_t column_number_;
        const std::string seperator_;
        size_t value_count_;
        std::stringstream ss_;

};


} //dyno
