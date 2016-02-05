#ifndef PARAMETERS_HH
#define PARAMETERS_HH

#include <deal.II/base/parameter_handler.h>


using namespace dealii;


namespace Parameters {

//     struct Materials {
//         double Young_modulus_matrix;
//         double Poisson_ratio_matrix;
//         double Young_modulus_fiber;
//         double Poisson_ratio_fiber;
//         double Fiber_volume_ratio;
//         unsigned int Reinforcement_material_id;
//         bool use_1d_fibers;
// 
//         static void declare_parameters(ParameterHandler &prm);
//         void parse_parameters(ParameterHandler &prm);
//     };

    struct BoundaryConditions {
        std::map<int, std::string > velocity;
        std::map<int, std::string > velocity_tangent;
        std::map<int, std::string > velocity_normal;
        std::map<int, std::string > velocity_slip;
        std::map<int, std::string > traction;

        static void declare_parameters(ParameterHandler &prm);
        void parse_parameters(ParameterHandler &prm);
    };

//     struct RightHandSide {
//         std::map<int, std::string> force;
// 
//         static void declare_parameters(ParameterHandler &prm);
//         void parse_parameters(ParameterHandler &prm);
//     };
// 
    struct IO {
//         std::string mesh_file;
        std::string output_file_base;

        static void declare_parameters(ParameterHandler &prm);
        void parse_parameters(ParameterHandler &prm);
    };


    struct AllParameters :
//             public Materials,
            public BoundaryConditions,
//             public RightHandSide,
            public IO
    {
        AllParameters(const std::string &input_file);
        static void declare_parameters(ParameterHandler &prm);
        void parse_parameters(ParameterHandler &prm);
    };

}




#endif
