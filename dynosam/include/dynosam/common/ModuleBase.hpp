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

#include "dynosam/utils/Variant.hpp"
#include "dynosam/common/Types.hpp"

#include "dynosam/common/ModulePacketVariants.hpp"

namespace dyno {

// //we expect the tempalted packets to be jsut the types (not pointers)
// //we provide a castInput function for IPacket or std::shared_ptr<const IPacket>
// //but expect the value contained in the packet to directly be MInput or MOutput
// //so if IPacket is a variant and contains a std::shared_ptr<MInput>, MInput shoudl be td::shared_ptr<MInput>
// template<typename IPacket, typename OPacket, typename MInput = IPacket, typename MOutput = OPacket>
// class VariantModuleBase {
// public:
//     using InputPacket = IPacket;
//     using OutputPacket = OPacket;
//     using ModuleInput = MInput;
//     using ModuleOutput = MOutput;

//     using InputPacketConstPtr = std::shared_ptr<const InputPacket>;
//     using OutputPacketConstPtr = std::shared_ptr<const OutputPacket>;

//     using ModuleInputConstPtr = std::shared_ptr<const ModuleInput>;
//     using ModuleOutputConstPtr = std::shared_ptr<const ModuleOutput>;

//     virtual ~VariantModuleBase() = default;

// protected:
//     constexpr static bool IsInputPacketVariant = is_variant_v<IPacket>;
//     constexpr static bool IsOutputPacketVariant = is_variant_v<OutputPacket>;

//     constexpr static bool IsModuleInputVariant = is_variant_v<ModuleInput>;
//     constexpr static bool IsModuleOutputVariant = is_variant_v<ModuleOutput>;

//      //either InputPacket is a variant or the InputPacket and ModuleInput are the same
//     //and no casting is required
//     static_assert(IsInputPacketVariant || std::is_same_v<ModuleInput,InputPacket>,
//         "InputPacket specified by IPacket is not a std::variant or ModuleInput and InputPacket are not the same type");

//     static_assert(IsOutputPacketVariant || std::is_same_v<ModuleOutput,OutputPacket>,
//         "OutputPacket specified by OPacket is not a std::variant or ModuleOutput and OutputPacket are not the same type");

//     // //if IsInputPacketVariant, then ModuleInput cannot also be a variant!!
//     static_assert(!(IsInputPacketVariant && IsModuleInputVariant),
//         "InputPacket is a variant, so ModuleInput cannot also be a variant, but a type within the variant");

//     static_assert(!(IsOutputPacketVariant && IsModuleOutputVariant),
//         "OutputPacket is a variant, so ModuleOutput cannot also be a variant, but a type within the variant");

//     static_assert(is_variant_member_v<ModuleInput, InputPacket> || std::is_same_v<ModuleInput,InputPacket>,
//         "If InputPacket is a variant, ModuleInput is not a type within the variant, or ModuleInput and InputPacket types are not the same");

//     static_assert(is_variant_member_v<ModuleOutput, OutputPacket> || std::is_same_v<ModuleOutput,OutputPacket>,
//         "If InputPacket is a variant, ModuleOutput is not a type within the variant, or ModuleOutput and InputPacket types are not the same");

// protected:
//     ModuleInput castInput(const InputPacketConstPtr& packet) const {
//         return castInput(*packet);
//     }
//     //expect InputPacket to be a pointer as its constructed by a pipeline
//     ModuleInput castInput(const InputPacket& packet) const {
//         if constexpr (IsInputPacketVariant) {
//             try {
//                 //packet should be a variant the ModuleOutput is a type within the std::variant
//                 //and the static asserts guarantee that ModuleInput is also not a variant
//                 return std::get<ModuleInput>(packet);
//             }
//             catch(const std::bad_variant_access& e) {
//                 throw std::runtime_error("Bad access");
//             }
//         }
//         else {
//            //if InputPacket is not a variant, then the static asserts guarantee that InputPacket == ModuleInput;
//            return packet;
//         }
//     }

// };

// //Module input and Module output could be poitners and in fact, in reality, they will be
// //as expect the dynosam pipeline to only work with pointers but we will specify this in another class
// template<typename IPacket, typename OPacket, typename MInput = IPacket, typename MOutput = OPacket>
// class VariantModule : public VariantModuleBase<IPacket, OPacket, MInput, MOutput> {

// public:
//     using Base = VariantModuleBase<IPacket, OPacket, MInput, MOutput>;
//     using ModuleInput = typename Base::ModuleInput;
//     using ModuleOutput = typename Base::ModuleOutput;
//     using InputPacketConstPtr = typename Base::InputPacketConstPtr;
//     using OutputPacketConstPtr = typename Base::OutputPacketConstPtr;

//     enum class State {
//         Boostrap = 0u, //! Initalize Frontend
//         Nominal = 1u //! Run Frontend
//     };

//     using SpinReturn = std::pair<State, OutputPacketConstPtr>;

//     OutputPacketConstPtr spinOnce(const InputPacketConstPtr& input) {
//         return boostrapSpin(Base::castInput(input)).second;
//         return nullptr;
//     }

// protected:
//     virtual void validateInput(const InputPacketConstPtr& input) const {};
//     virtual SpinReturn boostrapSpin(const ModuleInput& input) {
//         LOG(INFO) << input;
//         return SpinReturn{};
//     };
//     virtual SpinReturn nominalSpin(const ModuleInput& input) {};

// };



/**
 * @brief Defines some basic functions and behaviour for a module which will take and send input
 * to and from a pipeline.
 *
 * A module is intended to the core place where all the processing actually happens.
 * Like the pipeline, we expect INPUT and OUTPUT to have certain typedefs defined, partically
 * ::ConstPtr
 *
 *
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template<typename INPUT, typename OUTPUT>
class ModuleBase {

public:
    using InputConstPtr = typename INPUT::ConstPtr; //! ConstPtr to an INPUT type
    using OutputConstPtr = typename OUTPUT::ConstPtr; //! ConstPtr to an OUTPUT type
    using This = ModuleBase<INPUT, OUTPUT>;

    DYNO_POINTER_TYPEDEFS(This)

    enum class State {
        Boostrap = 0u, //! Initalize Frontend
        Nominal = 1u //! Run Frontend
    };

    using SpinReturn = std::pair<State, OutputConstPtr>;

    ModuleBase(const std::string& name) : name_(name), module_state_(State::Boostrap) {}
    virtual ~ModuleBase() = default;

    OutputConstPtr spinOnce(InputConstPtr input) {
        CHECK(input);
        validateInput(input);

        SpinReturn spin_return{State::Boostrap, nullptr};

        switch (module_state_)
            {
            case State::Boostrap: {
                spin_return = boostrapSpin(input);
            } break;
            case State::Nominal: {
                spin_return = nominalSpin(input);
            } break;
            default: {
                LOG(FATAL) << "Unrecognized state in module " << name_;
                break;
            }
        }

        module_state_ = spin_return.first;
        return spin_return.second;
    }

protected:
    /**
     * @brief Checks the state of the input before running any virtual spin (boostrap or nominal) functions.
     *
     * The function is intended to be treated as a "fail hard" case, and should throw an exception if the input is not valid
     *
     * @param input
     */
    virtual void validateInput(const InputConstPtr& input) const = 0;

    virtual SpinReturn boostrapSpin(InputConstPtr input) = 0;
    virtual SpinReturn nominalSpin(InputConstPtr input) = 0;



private:
    const std::string name_;
    std::atomic<State> module_state_;

};


} //dyno
