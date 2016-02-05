/* ---------------------------------------------------------------------
 * $Id: step-3.cc 30147 2013-07-24 09:28:41Z maier $
 *
 * Copyright (C) 1999 - 2013 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Authors: Wolfgang Bangerth, 1999,
 *          Guido Kanschat, 2011
 */


#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>
#include <boost/graph/buffer_concepts.hpp>

#include "parameters.hh"

using namespace dealii;

class StokesSlip
{
public:
  StokesSlip (const std::string &input_file);

  void run ();


private:
  void make_grid ();
  void setup_system ();
  void assemble_system ();
  void assemble_stress_rhs();
  double solve ();
  void output_shear_stress() const;
  void output_vtk () const;
  
  double phi_grad(double x, double sigma_bound, double sigma_growth)
  {
    // quadratic regularization
    return (fabs(x) < eps)?(x*(sigma_bound/eps+sigma_growth)):(sigma_bound*x/fabs(x)+x*sigma_growth); 
    
    // 4th order polynomial regularization
//     return (fabs(x) < eps)?(-pow(x/eps,3.)*sigma_bound*0.5+x*(1.5*sigma_bound/eps+2.*sigma_growth)):(sigma_bound*x/fabs(x)+x*2.*sigma_growth);
    
    // regularization by square root
//     return (sigma_bound/sqrt(x*x+eps*eps)+sigma_growth*2)*x;
  }
  
  double phi_hess(double x, double sigma_bound, double sigma_growth)
  { 
    // quadratic regularization
    return sigma_growth+((fabs(x) < eps)?(sigma_bound/eps):0);
    
    // 4th order polynomial regularization
//     return 2.*sigma_growth+((fabs(x) < eps)?(1.5*sigma_bound/eps*(1.-pow(x/eps,2.))):0);
    
    // regularization by square root
//     return sigma_bound*eps*eps/pow(x*x+eps*eps,1.5)+sigma_growth*2;
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

  Vector<double>       solution;
  Vector<double>       system_rhs;
  Vector<double>       stress_grad;
  SparseMatrix<double> stress_hess;
//   std::vector<double>    shear_stress;
//   std::vector<double>    slip_velocity;
//   std::vector<double>    shear_x;
  
  Parameters::AllParameters prm;
  
  const double eps = 1e-7;
};




StokesSlip::StokesSlip (const std::string &input_file)
  :
  fe(FE_Q<2>(3), 2, FE_Q<2>(2), 1),
  dof_handler(triangulation),
  quadrature_formula(QGauss<2>(4)),
  quad1d(QGauss<1>(2)),
  quad1d_output(QGaussLobatto<1>(2)),
  prm(input_file)
{}


void StokesSlip::make_grid ()
{
	GridGenerator::hyper_cube (triangulation, 0, 1, true);
	triangulation.refine_global (5);

	// deform the mesh according to a function alpha
    std::vector<bool> moved(triangulation.n_vertices(), false);
	for (Triangulation<2>::cell_iterator cell=triangulation.begin(); cell != triangulation.end(); ++cell)
	{
		for (unsigned int vid=0; vid<4; ++vid)
		{
          if (!moved[cell->vertex_index(vid)])
          {
			double x = cell->vertex(vid)(0);
			double y = cell->vertex(vid)(1);
			cell->vertex(vid)(1) += (1-y)*alpha(x);
            moved[cell->vertex_index(vid)] = true;
          }
		}
	}

	std::cout << "Number of active cells: "
		  << triangulation.n_active_cells()
		  << std::endl;
	std::cout << "Total number of cells: "
		  << triangulation.n_cells()
		  << std::endl;
}

/*
 * Boundary indicators:
 * 
 *        3
 *    +-------+
 *    |       |
 *   0|       |1
 *    |       |
 *    +-------+
 *        2
 */
void StokesSlip::setup_system ()
{
	dof_handler.distribute_dofs (fe);
	std::cout << "Number of degrees of freedom: "
		  << dof_handler.n_dofs()
		  << std::endl;

	CompressedSparsityPattern c_sparsity(dof_handler.n_dofs());
	DoFTools::make_sparsity_pattern (dof_handler, c_sparsity);
	sparsity_pattern.copy_from(c_sparsity);

	system_matrix.reinit (sparsity_pattern);

	solution.reinit (dof_handler.n_dofs());
	system_rhs.reinit (dof_handler.n_dofs());
	stress_grad.reinit (dof_handler.n_dofs());
    stress_hess.reinit (sparsity_pattern);

// 	// setup shear stress
// 	unsigned int n_shear_stress_dofs = 0;
// 	DoFHandler<2>::active_cell_iterator
// 	cell = dof_handler.begin_active(),
// 	endc = dof_handler.end();
// 	for (; cell!=endc; ++cell)
// 	{
// 		for (unsigned int face = 0; face < 4; ++face)
// 		{
// 			if (cell->at_boundary(face) && cell->face(face)->boundary_indicator() == 2)
// 			{
// 				n_shear_stress_dofs += quad1d_output.size();
// 			}
// 		}
// 	}
// 	shear_stress.resize(n_shear_stress_dofs, 0.);
//     slip_velocity.resize(n_shear_stress_dofs, 0.);
//     shear_x.resize(n_shear_stress_dofs, 0.);
}

void StokesSlip::assemble_system ()
{
	FEValues<2> fe_values (fe, quadrature_formula,
			update_values | update_gradients | update_JxW_values | update_quadrature_points);
	FEFaceValues<2> fe_face_values(fe, quad1d, update_values | update_normal_vectors | update_JxW_values | update_quadrature_points);
	
	const unsigned int   dofs_per_cell = fe.dofs_per_cell;
	const unsigned int   n_q_points    = quadrature_formula.size();

	FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
	Vector<double>       cell_rhs (dofs_per_cell);

	std::vector<Vector<double> > rhs_values (n_q_points, Vector<double>(2));

	std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

	const FEValuesExtractors::Vector velocities (0);
	const FEValuesExtractors::Scalar pressure (2);

	std::vector<Tensor<2,2> > grad_phi_u (dofs_per_cell);
	std::vector<double>       div_phi_u   (dofs_per_cell);
	std::vector<double>       phi_p       (dofs_per_cell);
	
	system_matrix = 0;
	system_rhs = 0;
    
	DoFHandler<2>::active_cell_iterator
	cell = dof_handler.begin_active(),
	endc = dof_handler.end();
	for (; cell!=endc; ++cell)
	{
		fe_values.reinit (cell);

		cell_matrix = 0;
		cell_rhs = 0;
		
		cell->get_dof_indices (local_dof_indices);

		for (unsigned int q=0; q<n_q_points; ++q)
		{
			for (unsigned int k=0; k<dofs_per_cell; ++k)
			{
				grad_phi_u[k] = fe_values[velocities].symmetric_gradient (k, q);
				div_phi_u[k]  = fe_values[velocities].divergence (k, q);
				phi_p[k]      = fe_values[pressure].value (k, q);
			}

			for (unsigned int i=0; i<dofs_per_cell; ++i)
			{
				for (unsigned int j=0; j<dofs_per_cell; ++j)
				{
					cell_matrix(i,j) += (scalar_product(grad_phi_u[i], grad_phi_u[j])
										  - div_phi_u[i] * phi_p[j]
										  - phi_p[i] * div_phi_u[j]
										)
										* fe_values.JxW(q);

				}

//				const unsigned int component_i =
//						fe.system_to_component_index(i).first;
				cell_rhs(i) += fe_values.shape_value(i,q) *
							  0*//rhs_values[q](component_i) *
							  fe_values.JxW(q);
			}
		}
		
		for (unsigned int face = 0; face < 4; ++face)
        {
          if (cell->at_boundary(face))
          {
            // prescribe normal component of velocity
            if (prm.velocity_normal.find(cell->face(face)->boundary_indicator()) != prm.velocity_normal.end())
            {
              fe_face_values.reinit(cell, face);
              std::vector<Vector<double> > bc_values(quad1d.size(), Vector<double>(1));
              FunctionParser<2> fp(1);
              fp.initialize("x,y", prm.velocity_normal[cell->face(face)->boundary_indicator()], {});
              fp.vector_value_list (fe_face_values.get_quadrature_points(), bc_values);

              for (unsigned int q_point=0; q_point<quad1d.size(); q_point++)
              {
                Point<2> normal = fe_face_values.normal_vector(q_point);
                
                for (unsigned int i=0; i<dofs_per_cell; i++)
                {
                  cell_rhs(i) += (fe_face_values[velocities].value(i,q_point)*normal) *
                        bc_values[q_point][0] /
                        eps *
                        fe_face_values.JxW(q_point);
                        
                  for (unsigned int j=0; j<dofs_per_cell; j++)
                    cell_matrix(i,j) += (fe_face_values[velocities].value(i,q_point)*normal) *
                        (fe_face_values[velocities].value(j,q_point)*normal) /
                        eps *
                        fe_face_values.JxW(q_point);
                }
              }
            }
            // prescribe zero normal component of velocity (slip b.c.)
            else if (prm.velocity_slip.find(cell->face(face)->boundary_indicator()) != prm.velocity_slip.end())
            {
              fe_face_values.reinit(cell, face);

              for (unsigned int q_point=0; q_point<quad1d.size(); q_point++)
              {
                Point<2> normal = fe_face_values.normal_vector(q_point);
                
                for (unsigned int i=0; i<dofs_per_cell; i++)
                {
                  for (unsigned int j=0; j<dofs_per_cell; j++)
                    cell_matrix(i,j) += (fe_face_values[velocities].value(i,q_point)*normal) *
                        (fe_face_values[velocities].value(j,q_point)*normal) /
                        eps *
                        fe_face_values.JxW(q_point);
                }
              }
            }
            // prescribe tangent part of velocity
            else if (prm.velocity_tangent.find(cell->face(face)->boundary_indicator()) != prm.velocity_tangent.end())
            {
              fe_face_values.reinit(cell, face);
              std::vector<Vector<double> > bc_values(quad1d.size(), Vector<double>(2));
              FunctionParser<2> fp(2);
              fp.initialize("x,y", prm.velocity_tangent[cell->face(face)->boundary_indicator()], {});
              fp.vector_value_list (fe_face_values.get_quadrature_points(), bc_values);

              for (unsigned int q_point=0; q_point<quad1d.size(); q_point++)
              {
                Point<2> normal = fe_face_values.normal_vector(q_point);
                Point<2> bc_tangent = { bc_values[q_point][0], bc_values[q_point][1] };
                bc_tangent -= (bc_tangent*normal)*normal;
                
                for (unsigned int i=0; i<dofs_per_cell; i++)
                {
                  cell_rhs(i) += (fe_face_values[velocities].value(i,q_point)-(fe_face_values[velocities].value(i,q_point)*normal)*normal) *
                        bc_tangent /
                        eps *
                        fe_face_values.JxW(q_point);
                        
                  for (unsigned int j=0; j<dofs_per_cell; j++)
                    cell_matrix(i,j) += (fe_face_values[velocities].value(i,q_point)-(fe_face_values[velocities].value(i,q_point)*normal)*normal) *
                        (fe_face_values[velocities].value(j,q_point)-(fe_face_values[velocities].value(j,q_point)*normal)*normal) /
                        eps *
                        fe_face_values.JxW(q_point);
                }
              }
            }
            // prescribe full velocity vector
            else if (prm.velocity.find(cell->face(face)->boundary_indicator()) != prm.velocity.end())
            {
              fe_face_values.reinit(cell, face);
              std::vector<Vector<double> > bc_values(quad1d.size(), Vector<double>(2));
              FunctionParser<2> fp(2);
              fp.initialize("x,y", prm.velocity[cell->face(face)->boundary_indicator()], {});
              fp.vector_value_list (fe_face_values.get_quadrature_points(), bc_values);

              for (unsigned int q_point=0; q_point<quad1d.size(); q_point++)
              {
                Point<2> bc_value = {bc_values[q_point][0],bc_values[q_point][1]};
                
                for (unsigned int i=0; i<dofs_per_cell; i++)
                {
                  cell_rhs(i) += fe_face_values[velocities].value(i,q_point)*bc_value / eps * fe_face_values.JxW(q_point);
                  
                  for (unsigned int j=0; j<dofs_per_cell; j++)
                    cell_matrix(i,j) += (fe_face_values[velocities].value(i,q_point)) *
                        (fe_face_values[velocities].value(j,q_point)) /
                        eps * fe_face_values.JxW(q_point);
                }
              }
            }
            // prescribe traction
            if (prm.traction.find(cell->face(face)->boundary_indicator()) != prm.traction.end())
            {
              fe_face_values.reinit(cell, face);
              std::vector<Vector<double> > bc_values(quad1d.size(), Vector<double>(2));
              FunctionParser<2> fp(2);
              fp.initialize("x,y", prm.traction[cell->face(face)->boundary_indicator()], {});
              fp.vector_value_list (fe_face_values.get_quadrature_points(), bc_values);

              for (unsigned int q_point=0; q_point<quad1d.size(); q_point++)
              {
                Point<2> traction = {bc_values[q_point][0], bc_values[q_point][1]};
                
                for (unsigned int i=0; i<dofs_per_cell; i++)
                {
                  cell_rhs(i) += (fe_face_values[velocities].value(i,q_point)*traction) *
                        fe_face_values.JxW(q_point);
                }
              }
            }
          }
        }
		
        system_matrix.add(local_dof_indices, local_dof_indices, cell_matrix);
        system_rhs.add(local_dof_indices, cell_rhs);
	}
}


void StokesSlip::assemble_stress_rhs ()
{
	FEFaceValues<2> fe_face_values(fe, quad1d, update_values | update_JxW_values | update_normal_vectors | update_quadrature_points);
	
	const unsigned int   dofs_per_cell = fe.dofs_per_cell;

	Vector<double>       cell_grad (dofs_per_cell);
    FullMatrix<double>   cell_hess(dofs_per_cell,dofs_per_cell);

	std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

	const FEValuesExtractors::Vector velocities (0);
	const FEValuesExtractors::Scalar pressure (2);
	
	stress_grad = 0;
    stress_hess = 0;

	DoFHandler<2>::active_cell_iterator
	cell = dof_handler.begin_active(),
	endc = dof_handler.end();
	for (; cell!=endc; ++cell)
	{
		cell_grad = 0;
        cell_hess = 0;
        
        cell->get_dof_indices (local_dof_indices);
		
		for (unsigned int face = 0; face < 4; ++face)
		{
			if (cell->at_boundary(face) && prm.velocity_slip.find(cell->face(face)->boundary_indicator()) != prm.velocity_slip.end())
			{
				fe_face_values.reinit(cell, face);
                std::vector<Vector<double> > slip_values(quad1d.size(), Vector<double>(2));
                FunctionParser<2> fp(2);
                fp.initialize("x,y", prm.velocity_slip[cell->face(face)->boundary_indicator()], {});
                fp.vector_value_list (fe_face_values.get_quadrature_points(), slip_values);

				for (unsigned int q_point=0; q_point<quad1d.size(); q_point++)
				{
                    Point<2> u({0, 0});
                    for (unsigned int i=0; i<dofs_per_cell; i++)
                      u += solution[local_dof_indices[i]]*fe_face_values[velocities].value(i,q_point);
                    
                    Point<2> normal = fe_face_values.normal_vector(q_point);
                    Point<2> tangent = {-normal[1], normal[0]};
                    double u_tau = u*tangent;
					
					for (unsigned int i=0; i<dofs_per_cell; i++)
					{
						cell_grad(i) += (fe_face_values[velocities].value(i,q_point) * tangent) *
							phi_grad(u_tau, slip_values[q_point][0], slip_values[q_point][1]) *
							fe_face_values.JxW(q_point);
                        
                        for (unsigned int j=0; j<dofs_per_cell; j++)
                            cell_hess(i,j) += (fe_face_values[velocities].value(i,q_point)*tangent) *
                                (fe_face_values[velocities].value(j,q_point)*tangent) *
                                phi_hess(u_tau, slip_values[q_point][0], slip_values[q_point][1]) *
                                fe_face_values.JxW(q_point);
					}
				}
			}
		}
		
		stress_grad.add(local_dof_indices, cell_grad);
        stress_hess.add(local_dof_indices,local_dof_indices, cell_hess);
	}
}



void StokesSlip::output_shear_stress() const
{
  FEFaceValues<2> fe_face_values(fe, quad1d_output, update_values | update_gradients | update_normal_vectors | update_quadrature_points);
  const FEValuesExtractors::Vector velocities (0);
  std::vector<types::global_dof_index> local_dof_indices (fe.dofs_per_cell);
  double shear_stress;
  double slip_velocity;

  std::ofstream f(prm.output_file_base + ".dat");
  f << "# X Y Shear_stress U_tau" << std::endl;
  
  DoFHandler<2>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();
  for (; cell!=endc; ++cell)
  {
    for (unsigned int face = 0; face < 4; ++face)
    {
      if (cell->at_boundary(face) && prm.velocity_slip.find(cell->face(face)->boundary_indicator()) != prm.velocity_slip.end())
      {
        fe_face_values.reinit(cell, face);
        cell->get_dof_indices (local_dof_indices);

        for (unsigned int q_point=0; q_point<quad1d_output.size(); q_point++)
        {
          shear_stress = 0;
          slip_velocity = 0;
          Point<2> normal = fe_face_values.normal_vector(q_point);
          Point<2> tangent = {-normal[1], normal[0]};
          
          for (unsigned int i=0; i<fe.dofs_per_cell; i++)
          {
            Point<2> Dphi_n = fe_face_values[velocities].symmetric_gradient(i,q_point)*normal;
            shear_stress += solution[local_dof_indices[i]]*(Dphi_n*tangent);
            slip_velocity += solution[local_dof_indices[i]]*fe_face_values[velocities].value(i,q_point)*tangent;
          }
          f << fe_face_values.quadrature_point(q_point) << " " << shear_stress << " " << slip_velocity << std::endl;
        }
      }
    }
  }
}




void StokesSlip::output_vtk () const
{
	std::vector<std::string> solution_names (2, "velocity");
	solution_names.push_back ("pressure");

	std::vector<DataComponentInterpretation::DataComponentInterpretation>
	data_component_interpretation(2, DataComponentInterpretation::component_is_part_of_vector);
	data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);

	DataOut<2> data_out;
	data_out.attach_dof_handler(dof_handler);
	data_out.add_data_vector (solution, solution_names,
				  DataOut<2>::type_dof_data,
				  data_component_interpretation);
	data_out.build_patches ();

	std::ofstream output (prm.output_file_base + ".vtk");
	data_out.write_vtk (output);
}


double StokesSlip::solve ()
{
    SparseDirectUMFPACK solver;
    
    SparseMatrix<double> mat(system_matrix.get_sparsity_pattern());
    mat.copy_from(system_matrix);
    mat.add(1., stress_hess);
    
    Vector<double> rhs(solution.size());
    system_matrix.vmult(rhs, solution);
    rhs *= -1;
    rhs.add(-1, stress_grad);
    rhs.add(1., system_rhs);
    

    solver.solve(mat, rhs);
    
    solution.add(1.,rhs);

    return rhs.l2_norm()/system_rhs.l2_norm();
}


void StokesSlip::run ()
{
  double delta;
  unsigned int iter = 0;
  
  make_grid ();
  setup_system();
  assemble_system();
  do
  {
    assemble_stress_rhs();
    delta = solve ();
    iter++;
    std::cout << "Iter. " << iter << " Delta: " << delta << std::endl;
  } while ((delta > 1e-8) && iter < 30);
  
  output_shear_stress();
  
  output_vtk ();
}


int main (int argc, char **argv)
{
	StokesSlip stokes_problem(argv[1]);
	stokes_problem.run ();

	return 0;
}
