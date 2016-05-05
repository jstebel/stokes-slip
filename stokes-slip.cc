#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/grid/grid_in.h>
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
#include "interpolation.h" // alglib
#include "stdafx.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "stokes-slip.hh"

using namespace dealii;





StokesSlip::StokesSlip (const std::string &input_file)
  :
  fe(FE_Q<2>(3), 2, FE_Q<2>(2), 1),
  dof_handler(triangulation),
  quadrature_formula(QGauss<2>(4)),
  quad1d(QGauss<1>(2)),
  quad1d_output(QGaussLobatto<1>(2)),
  prm_(input_file)
{}

double bezier(const alglib::real_1d_array &a, double x)
{
    unsigned int n = a.length();
    if (x == 1) return a[n];
    
    double b = 0;
    double term = pow(1.-x,n);
    for (unsigned int i=0; i<n; i++)
    {
      if (i > 0)
        term *= x*(n-i+1)/((1.-x)*i);
      b += a[i]*term;
    }
    return b;
}


void StokesSlip::make_grid (const alglib::real_1d_array &x)
{
    std::cout << "x = ";
    for (unsigned int i=0; i<prm_.np; ++i) std::cout << x[i] << " ";
    std::cout << std::endl;
    
	GridIn<2> grid_in;
	grid_in.attach_triangulation(triangulation);

	std::ifstream input_file(prm_.mesh_file);
	Assert (input_file, ExcFileNotOpen(prm_.mesh_file.c_str()));

	std::cout << "* Read mesh file '" << prm_.mesh_file << "'"<< std::endl;
    triangulation.clear();
	grid_in.read_msh(input_file);
    
    double l = (prm_.p2-prm_.p1).norm();
    Tensor<1,2> t=(prm_.p2-prm_.p1)/l, n({-t[1], t[0]});
//     std::vector<double> X(prm_.np);
//     for (unsigned int i=0; i<prm_.np; ++i)
//       X[i] = double(i)/(prm_.np-1);
//     alglib::real_1d_array AX;
//     AX.setcontent(prm_.np, &(X[0]));
//     alglib::spline1dinterpolant spline;
//     alglib::spline1dbuildcubic(AX, x, X.size(), 2,0.0,2,0.0, spline);
    // deform the mesh according to a function alpha
    std::vector<bool> moved(triangulation.n_vertices(), false);
    for (Triangulation<2>::cell_iterator cell=triangulation.begin(); cell != triangulation.end(); ++cell)
    {
        for (unsigned int vid=0; vid<4; ++vid)
        {
          if (!moved[cell->vertex_index(vid)])
          {
            Point<2> p = cell->vertex(vid);
            double pn = (p-prm_.p1)*n, pt = (p-prm_.p1)*t;
            if (pn >= 0 & pn <= prm_.height & pt >= 0 & pt <= l)
              cell->vertex(vid) += n*(1-(p-prm_.p1)*n/prm_.height)*bezier(x, (p-prm_.p1)*t/l);//alglib::spline1dcalc(spline,(p-prm_.p1)*t/l);
            moved[cell->vertex_index(vid)] = true;
          }
        }
    }
    
//     // deformation of mesh using Laplace equation
//     std::map<unsigned int, Point<2> > deformed_points;
//     for (auto cell=triangulation.begin_face(); cell != triangulation.end_face(); ++cell)
//     {
//       if (!cell->at_boundary()) continue;
//       if (cell->boundary_indicator() == 5) continue;
//       for (int vid=0; vid<2; ++vid)
//       {
//         if (deformed_points.find(cell->vertex_index(vid)) == deformed_points.end())
//         {
//           Point<2> p = cell->vertex(vid);
//           if (cell->boundary_indicator() == 4) p[0] += p[1];
//           deformed_points[cell->vertex_index(vid)] = p;
//         }
//       }
//     }
//     ConstantFunction<2> c(1000);
//     GridTools::laplace_transform(deformed_points, triangulation, &c);
    
/*	
	GridGenerator::hyper_cube (triangulation, 0, 1, true);
	triangulation.refine_global (7);

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
*/
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

    std::vector<Tensor<1,2> > u(dofs_per_cell);
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
                u[k] = fe_values[velocities].value(k,q);
                                if (prm_.sym_grad)
				  grad_phi_u[k] = fe_values[velocities].symmetric_gradient(k, q)*sqrt(2.);
				else
				  grad_phi_u[k] = fe_values[velocities].gradient(k, q);
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
			}
		}
		
		// Assembling the right hand side due to volume force
        if (prm_.force.find(cell->material_id()) != prm_.force.end())
        {
            FunctionParser<2> fp(2);
            fp.initialize("x,y", prm_.force[cell->material_id()], {});
            fp.vector_value_list (fe_values.get_quadrature_points(), rhs_values);
            for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
            {
              for (unsigned int i=0; i<dofs_per_cell; ++i)
              {
                Tensor<1,2> rhs({rhs_values[q_point][0], rhs_values[q_point][1]});
                cell_rhs(i) += fe_values[velocities].value(i,q_point) *
                    rhs *
                    fe_values.JxW(q_point);
              }
            }
        }
		
		for (unsigned int face = 0; face < 4; ++face)
        {
          if (cell->at_boundary(face))
          {
            // prescribe normal component of velocity
            if (prm_.velocity_normal.find(cell->face(face)->boundary_indicator()) != prm_.velocity_normal.end())
            {
              fe_face_values.reinit(cell, face);
              std::vector<Vector<double> > bc_values(quad1d.size(), Vector<double>(1));
              FunctionParser<2> fp(1);
              fp.initialize("x,y", prm_.velocity_normal[cell->face(face)->boundary_indicator()], {});
              fp.vector_value_list (fe_face_values.get_quadrature_points(), bc_values);

              for (unsigned int q_point=0; q_point<quad1d.size(); q_point++)
              {
                Tensor<1,2> normal = fe_face_values.normal_vector(q_point);
                
                for (unsigned int i=0; i<dofs_per_cell; i++)
                {
                  cell_rhs(i) += (fe_face_values[velocities].value(i,q_point)*normal) *
                        bc_values[q_point][0] /
                        prm_.epsilon *
                        fe_face_values.JxW(q_point);
                        
                  for (unsigned int j=0; j<dofs_per_cell; j++)
                    cell_matrix(i,j) += (fe_face_values[velocities].value(i,q_point)*normal) *
                        (fe_face_values[velocities].value(j,q_point)*normal) /
                        prm_.epsilon *
                        fe_face_values.JxW(q_point);
                }
              }
            }
            // prescribe zero normal component of velocity (slip b.c.)
            else if (prm_.velocity_slip.find(cell->face(face)->boundary_indicator()) != prm_.velocity_slip.end())
            {
              fe_face_values.reinit(cell, face);

              for (unsigned int q_point=0; q_point<quad1d.size(); q_point++)
              {
                Tensor<1,2> normal = fe_face_values.normal_vector(q_point);
                
                for (unsigned int i=0; i<dofs_per_cell; i++)
                {
                  for (unsigned int j=0; j<dofs_per_cell; j++)
                    cell_matrix(i,j) += (fe_face_values[velocities].value(i,q_point)*normal) *
                        (fe_face_values[velocities].value(j,q_point)*normal) /
                        prm_.epsilon *
                        fe_face_values.JxW(q_point);
                }
              }
            }
            // prescribe tangent part of velocity
            else if (prm_.velocity_tangent.find(cell->face(face)->boundary_indicator()) != prm_.velocity_tangent.end())
            {
              fe_face_values.reinit(cell, face);
              std::vector<Vector<double> > bc_values(quad1d.size(), Vector<double>(2));
              FunctionParser<2> fp(2);
              fp.initialize("x,y", prm_.velocity_tangent[cell->face(face)->boundary_indicator()], {});
              fp.vector_value_list (fe_face_values.get_quadrature_points(), bc_values);

              for (unsigned int q_point=0; q_point<quad1d.size(); q_point++)
              {
                Tensor<1,2> normal = fe_face_values.normal_vector(q_point);
                Tensor<1,2> bc_tangent({ bc_values[q_point][0], bc_values[q_point][1] });
                bc_tangent -= (bc_tangent*normal)*normal;
                
                for (unsigned int i=0; i<dofs_per_cell; i++)
                {
                  cell_rhs(i) += (fe_face_values[velocities].value(i,q_point)-(fe_face_values[velocities].value(i,q_point)*normal)*normal) *
                        bc_tangent /
                        prm_.epsilon *
                        fe_face_values.JxW(q_point);
                        
                  for (unsigned int j=0; j<dofs_per_cell; j++)
                    cell_matrix(i,j) += (fe_face_values[velocities].value(i,q_point)-(fe_face_values[velocities].value(i,q_point)*normal)*normal) *
                        (fe_face_values[velocities].value(j,q_point)-(fe_face_values[velocities].value(j,q_point)*normal)*normal) /
                        prm_.epsilon *
                        fe_face_values.JxW(q_point);
                }
              }
            }
            // prescribe full velocity vector
            else if (prm_.velocity.find(cell->face(face)->boundary_indicator()) != prm_.velocity.end())
            {
              fe_face_values.reinit(cell, face);
              std::vector<Vector<double> > bc_values(quad1d.size(), Vector<double>(2));
              FunctionParser<2> fp(2);
              fp.initialize("x,y", prm_.velocity[cell->face(face)->boundary_indicator()], {});
              fp.vector_value_list (fe_face_values.get_quadrature_points(), bc_values);

              for (unsigned int q_point=0; q_point<quad1d.size(); q_point++)
              {
                Tensor<1,2> bc_value({bc_values[q_point][0],bc_values[q_point][1]});
                
                for (unsigned int i=0; i<dofs_per_cell; i++)
                {
                  cell_rhs(i) += fe_face_values[velocities].value(i,q_point)*bc_value / prm_.epsilon * fe_face_values.JxW(q_point);
                  
                  for (unsigned int j=0; j<dofs_per_cell; j++)
                    cell_matrix(i,j) += (fe_face_values[velocities].value(i,q_point)) *
                        (fe_face_values[velocities].value(j,q_point)) /
                        prm_.epsilon * fe_face_values.JxW(q_point);
                }
              }
            }
            // prescribe traction
            if (prm_.traction.find(cell->face(face)->boundary_indicator()) != prm_.traction.end())
            {
              fe_face_values.reinit(cell, face);
              std::vector<Vector<double> > bc_values(quad1d.size(), Vector<double>(2));
              FunctionParser<2> fp(2);
              fp.initialize("x,y", prm_.traction[cell->face(face)->boundary_indicator()], {});
              fp.vector_value_list (fe_face_values.get_quadrature_points(), bc_values);

              for (unsigned int q_point=0; q_point<quad1d.size(); q_point++)
              {
                Tensor<1,2> traction({bc_values[q_point][0], bc_values[q_point][1]});
                
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
			if (cell->at_boundary(face) && prm_.velocity_slip.find(cell->face(face)->boundary_indicator()) != prm_.velocity_slip.end())
			{
				fe_face_values.reinit(cell, face);
                std::vector<Vector<double> > slip_values(quad1d.size(), Vector<double>(2));
                FunctionParser<2> fp(2);
                fp.initialize("x,y", prm_.velocity_slip[cell->face(face)->boundary_indicator()], {});
                fp.vector_value_list (fe_face_values.get_quadrature_points(), slip_values);

				for (unsigned int q_point=0; q_point<quad1d.size(); q_point++)
				{
                    Tensor<1,2> u({0, 0});
                    for (unsigned int i=0; i<dofs_per_cell; i++)
                      u += solution[local_dof_indices[i]]*fe_face_values[velocities].value(i,q_point);
                    
                    Tensor<1,2> normal = fe_face_values.normal_vector(q_point);
                    Tensor<1,2> tangent({-normal[1], normal[0]});
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

  std::ofstream f(prm_.output_file_base + ".dat");
  f << "# X Y Shear_stress U_tau" << std::endl;
  
  DoFHandler<2>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();
  for (; cell!=endc; ++cell)
  {
    for (unsigned int face = 0; face < 4; ++face)
    {
      if (cell->at_boundary(face) && prm_.velocity_slip.find(cell->face(face)->boundary_indicator()) != prm_.velocity_slip.end())
      {
        fe_face_values.reinit(cell, face);
        cell->get_dof_indices (local_dof_indices);

        for (unsigned int q_point=0; q_point<quad1d_output.size(); q_point++)
        {
          shear_stress = 0;
          slip_velocity = 0;
          Tensor<1,2> normal = fe_face_values.normal_vector(q_point);
          Tensor<1,2> tangent({-normal[1], normal[0]});
          
          for (unsigned int i=0; i<fe.dofs_per_cell; i++)
          {
            Tensor<1,2> Dphi_n;
            if (prm_.sym_grad)
              Dphi_n = fe_face_values[velocities].symmetric_gradient(i,q_point)*2*normal;
            else
              Dphi_n = fe_face_values[velocities].gradient(i,q_point)*normal;
            shear_stress += solution[local_dof_indices[i]]*(Dphi_n*tangent);
            slip_velocity += solution[local_dof_indices[i]]*fe_face_values[velocities].value(i,q_point)*tangent;
          }
          f << fe_face_values.quadrature_point(q_point) << " " << shear_stress << " " << slip_velocity << std::endl;
        }
      }
    }
  }
}


double StokesSlip::output_cost_function()
{
  FEValues<2> fe_values (fe, quadrature_formula,
            update_values | update_gradients | update_JxW_values | update_quadrature_points);
  FEFaceValues<2> fe_face_values(fe, quad1d_output, update_values | update_gradients | update_normal_vectors | update_quadrature_points | update_JxW_values);
  const FEValuesExtractors::Vector velocities (0);
  const FEValuesExtractors::Scalar pressure (2);
  std::vector<types::global_dof_index> local_dof_indices (fe.dofs_per_cell);
  double st, sn, ut, un;
  double cost = 0;

  DoFHandler<2>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();
  for (; cell!=endc; ++cell)
  {
    cell->get_dof_indices (local_dof_indices);
    
    if (prm_.volume_integral.find(cell->material_id()) != prm_.volume_integral.end())
    {
      fe_values.reinit(cell);
      
      for (unsigned int q_point=0; q_point<quadrature_formula.size(); q_point++)
      {
        double ux = 0, uy = 0, pr = 0, dxux = 0;
        
        for (unsigned int i=0; i<fe.dofs_per_cell; i++)
        {
          ux += solution[local_dof_indices[i]]*fe_values[velocities].value(i,q_point)[0];
          uy += solution[local_dof_indices[i]]*fe_values[velocities].value(i,q_point)[1];
          pr  += solution[local_dof_indices[i]]*fe_values[pressure].value(i,q_point);
          dxux += solution[local_dof_indices[i]]*fe_values[velocities].gradient(i,q_point)[0][0];
        }

        FunctionParser<2> fp(1);
        fp.initialize("x,y", prm_.volume_integral[cell->material_id()], {{"ux",ux}, {"uy",uy}, {"p",pr}, {"dxux",dxux}});
        
        Point<2> p({fe_values.quadrature_point(q_point)[0], fe_values.quadrature_point(q_point)[1]});
        cost += fp.value(p)*fe_values.JxW(q_point);
      }
    }
    
    for (unsigned int face = 0; face < 4; ++face)
    {
      if (cell->at_boundary(face) && prm_.boundary_integral.find(cell->face(face)->boundary_indicator()) != prm_.boundary_integral.end())
      {
        fe_face_values.reinit(cell, face);
        
        for (unsigned int q_point=0; q_point<quad1d_output.size(); q_point++)
        {
          st = sn = ut = un = 0;
          Tensor<1,2> normal = fe_face_values.normal_vector(q_point);
          Tensor<1,2> tangent({-normal[1], normal[0]});
          
          for (unsigned int i=0; i<fe.dofs_per_cell; i++)
          {
            Tensor<1,2> Dphi_n;
            if (prm_.sym_grad)
              Dphi_n = fe_face_values[velocities].symmetric_gradient(i,q_point)*2*normal;
            else
              Dphi_n = fe_face_values[velocities].gradient(i,q_point)*normal;
            st += solution[local_dof_indices[i]]*(Dphi_n*tangent);
            sn += solution[local_dof_indices[i]]*(Dphi_n*normal);
            ut += solution[local_dof_indices[i]]*fe_face_values[velocities].value(i,q_point)*tangent;
            un += solution[local_dof_indices[i]]*fe_face_values[velocities].value(i,q_point)*normal;
          }
          
          FunctionParser<2> fp(1);
          fp.initialize("x,y", prm_.boundary_integral[cell->face(face)->boundary_indicator()], {{"ut",ut}, {"un",un}, {"st",st}, {"sn",sn}});
          
          Point<2> p({fe_face_values.quadrature_point(q_point)[0], fe_face_values.quadrature_point(q_point)[1]});
          cost += fp.value(p)*fe_face_values.JxW(q_point);
        }
      }
    }
  }
  
  std::cout << "Cost function = " << cost << std::endl;
  return cost;
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

	std::ofstream output (prm_.output_file_base + ".vtk");
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

    return rhs.l2_norm();
}


double StokesSlip::run (const alglib::real_1d_array &x)
{
  double delta;
  unsigned int iter = 0;
  
  make_grid (x);
  setup_system();
  assemble_system();
  double system_rhs_norm = system_rhs.l2_norm();
  
  printf("\nIter Abs_residual Rel_residual\n");
  do
  {
    assemble_stress_rhs();
    delta = solve ();
    iter++;
    printf("%4d %12g %12g\n", iter, delta, delta/system_rhs_norm);
  } while (delta > std::max(prm_.a_tol, prm_.r_tol*system_rhs_norm) && iter < prm_.max_iter);
  
  printf("\n");
  
  if (delta <= prm_.a_tol) printf("Absolute tolerance reached.\n");
  if (delta <= prm_.r_tol*system_rhs_norm) printf("Relative tolerance reached.\n");
  if (iter  >= prm_.max_iter) printf("Maximal number of iterations reached.\n");
  
  output_vtk ();
  output_shear_stress();
  return output_cost_function();
}


