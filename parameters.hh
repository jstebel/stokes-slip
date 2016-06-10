#ifndef PARAMETERS_HH
#define PARAMETERS_HH

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>


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

    struct BulkParameters {
        std::map<int, std::string> force;
        bool sym_grad;

        static void declare_parameters(ParameterHandler &prm);
        void parse_parameters(ParameterHandler &prm);
    };

    struct Numerics {
      double epsilon;
      double a_tol;
      double r_tol;
      unsigned int max_iter;
      
      static void declare_parameters(ParameterHandler &prm);
      void parse_parameters(ParameterHandler &prm);
    };
    
    struct IO {
        unsigned int dim;
        std::string mesh_file;
        std::string output_file_base;

        static void declare_parameters(ParameterHandler &prm);
        void parse_parameters(ParameterHandler &prm);
    };
    
    struct CostFunction {
      std::map<int, std::string> boundary_integral;
      std::map<int, std::string> volume_integral;
      
      static void declare_parameters(ParameterHandler &prm);
      void parse_parameters(ParameterHandler &prm);
    };
    
    struct ShapeOptimization {
      Point<3> p1;
      Point<3> p2;
      Point<3> ptop;
      double height;
      unsigned int np;
      unsigned int maxit;
      std::vector<double> init_guess;
      std::string f_max;
      std::string f_min;
      double g_max;
      double h_max;
      std::string logfile;
      
      static void declare_parameters(ParameterHandler &prm);
      void parse_parameters(ParameterHandler &prm);
    };


    struct AllParameters :
//             public Materials,
            public BoundaryConditions,
            public BulkParameters,
            public CostFunction,
            public Numerics,
            public IO,
            ShapeOptimization
    {
        AllParameters(const std::string &input_file);
        static void declare_parameters(ParameterHandler &prm);
        void parse_parameters(ParameterHandler &prm);
        void print(std::ostream &out);
        
        ParameterHandler prm;
    };

}




#endif
