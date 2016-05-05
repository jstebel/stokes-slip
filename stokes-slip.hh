#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include "parameters.hh"

using namespace dealii;

class StokesSlip
{
public:
  StokesSlip (const std::string &input_file);

  double run (const alglib::real_1d_array &x);
  
  const Parameters::AllParameters &prm() const { return prm_; }


private:
  void make_grid (const alglib::real_1d_array &x);
  void setup_system ();
  void assemble_system ();
  void assemble_stress_rhs();
  double solve ();
  void output_shear_stress() const;
  double output_cost_function();
  void output_vtk () const;
  
  double phi_grad(double x, double sigma_bound, double sigma_growth)
  {
    // quadratic regularization
    return (fabs(x) < prm_.epsilon)?(x*(sigma_bound/prm_.epsilon+sigma_growth)):(sigma_bound*x/fabs(x)+x*sigma_growth); 
    
    // 4th order polynomial regularization
//     return (fabs(x) < eps)?(-pow(x/eps,3.)*sigma_bound*0.5+x*(1.5*sigma_bound/eps+sigma_growth)):(sigma_bound*x/fabs(x)+x*sigma_growth);
    
    // regularization by square root
//     return (sigma_bound/sqrt(x*x+eps*eps)+sigma_growth*2)*x;
  }
  
  double phi_hess(double x, double sigma_bound, double sigma_growth)
  { 
    // quadratic regularization
    return sigma_growth+((fabs(x) < prm_.epsilon)?(sigma_bound/prm_.epsilon):0);
    
    // 4th order polynomial regularization
//     return sigma_growth+((fabs(x) < eps)?(1.5*sigma_bound/eps*(1.-pow(x/eps,2.))):0);
    
    // regularization by square root
//     return sigma_bound*eps*eps/pow(x*x+eps*eps,1.5)+sigma_growth;
  }

  double alpha(double x)
  {
    return 0.2*exp(-10*pow(0.5-x,2));
//            0.5*x;
  }
  
  double alpha_p(double x)
  {
    return 4*(0.5-x)*exp(-10*pow(0.5-x,2));
//            0.5;
  }

  Triangulation<2>     triangulation;
  FESystem<2>          fe;
  DoFHandler<2>        dof_handler;
  Quadrature<2>        quadrature_formula;
  Quadrature<1>        quad1d;
  Quadrature<1>        quad1d_output;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;
  SparseMatrix<double> stress_hess;

  Vector<double>       solution;
  Vector<double>       system_rhs;
  Vector<double>       stress_grad;
  
  Parameters::AllParameters prm_;
};




