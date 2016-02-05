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

#include <deal.II/fe/fe_face.h>
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

#include <nlopt.h>

using namespace dealii;

class StokesSlip
{
public:
  StokesSlip ();
  ~StokesSlip();
  
  double evaluate_fg(const double *x, double *grad);
  unsigned int size()
  { return p_sol.size() + s_sol.size(); }
  double *lower_bounds();
  double *upper_bounds();
  
  void output_results(double *x);
  

private:
  void make_grid ();
  void setup_system ();
  void assemble_system ();
  void solve(Vector<double> &rhs, bool distribute);


  Triangulation<2>     triangulation;
  FESystem<2>          fe;
  FESystem<2>          fe_v;
  FE_Q<2>              fe_p;
  FE_FaceQ<2>          fe_s;
  DoFHandler<2>        dh;
  DoFHandler<2>        dh_v;
  DoFHandler<2>        dh_p;
  DoFHandler<2>        dh_s;
  Quadrature<1>        quad1d;

  BlockSparsityPattern bsp;
  BlockSparseMatrix<double> bsm;
  BlockSparsityPattern      vv_sparsity;
  SparsityPattern      *vp_sparsity;
  SparsityPattern      *sv_sparsity;
  BlockSparseMatrix<double> vv_mat; // A
  SparseMatrix<double> *vp_mat; // B
  SparseMatrix<double> *sv_mat; // C
//   SparseMatrix<double> sn_mat; // N
  ConstraintMatrix     constraints;

  BlockVector<double>  bv;
  Vector<double>       v_sol;
  Vector<double>       p_sol;
  Vector<double>       s_sol;
  BlockVector<double>  v_rhs; // f
  
  std::vector<types::global_dof_index> s_dof_map;
  std::vector<int> s_node_to_dof;
  std::vector<Point<2> > s_normal;
  
  double *lb, *ub;
  
  const double sigma_bound = 1e3;
  const unsigned int slip_boundary_id = 2;
  
  const unsigned int mesh_refinement = 5;
};


StokesSlip stokes_problem;


















StokesSlip::StokesSlip ()
  :
  fe(FE_Q<2>(2), 2, FE_Q<2>(1), 1, FE_FaceQ<2>(0), 1),
  fe_v(FE_Q<2>(2), 2),
  fe_p(1),
  fe_s(0),
  dh(triangulation),
  dh_v(triangulation),
  dh_p(triangulation),
  dh_s(triangulation),
  quad1d(QGauss<1>(2)),
  lb(nullptr),
  ub(nullptr)
{
  make_grid();
  setup_system();
  assemble_system();
}


StokesSlip::~StokesSlip()
{
  if (lb != nullptr) delete[] lb;
  if (ub != nullptr) delete[] ub;
}


void StokesSlip::make_grid ()
{
  GridGenerator::hyper_cube (triangulation, 0, 1, true);
  triangulation.refine_global (mesh_refinement);

  // deform the mesh according to a function alpha
  FunctionParser<1> alpha;
  alpha.initialize("x", "0.2*exp(-(0.5-x)^2*100)", { { "pi", 3.14 } });
//alpha.initialize("x", "0.5*x", { { "pi", 3.14 } });
  for (Triangulation<2>::cell_iterator cell=triangulation.begin_active(); cell != triangulation.end(); ++cell)
  {
    for (unsigned int vid=0; vid<2; ++vid)
    {
      double x = cell->vertex(vid)(0);
      double y = cell->vertex(vid)(1);
      Point<1> p(x);
      cell->vertex(vid)(1) += (1-y)*alpha.value(p);
    }
  }
  
  s_node_to_dof.resize(triangulation.n_vertices(), -1.);
  unsigned int last_index = 0;
  std::vector<unsigned int> node_duplicity(triangulation.n_vertices(), 0.);
  FEFaceValues<2> ffv(fe_v, quad1d, update_normal_vectors);
  for (Triangulation<2>::cell_iterator cell=triangulation.begin_active(); cell != triangulation.end(); ++cell)
  {
    for (unsigned int face = 0; face < 4; ++face)
    {
      if (cell->at_boundary(face) && cell->face(face)->boundary_indicator() == slip_boundary_id)
      {
	ffv.reinit(cell, face);
	for (unsigned int node=0; node<2; node++)
	{
	  unsigned int node_id = cell->face(face)->vertex_index(node);
	  if (s_node_to_dof[node_id] == -1)
	  {
	    s_node_to_dof[node_id] = last_index;
	    node_duplicity[node_id]++;
	    last_index++;
	  }
	}
      }
    }
  }
  s_normal.resize(last_index);
  for (Triangulation<2>::cell_iterator cell=triangulation.begin_active(); cell != triangulation.end(); ++cell)
  {
    for (unsigned int face = 0; face < 4; ++face)
    {
      if (cell->at_boundary(face) && cell->face(face)->boundary_indicator() == slip_boundary_id)
      {
	ffv.reinit(cell, face);
	for (unsigned int node=0; node<2; node++)
	{
	  unsigned int node_id = cell->face(face)->vertex_index(node);
	  s_normal[s_node_to_dof[node_id]] += ffv.normal_vector(0)/node_duplicity[node_id];
	}
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
	dh.distribute_dofs(fe);
	dh_v.distribute_dofs(fe_v);
	dh_p.distribute_dofs(fe_p);
	dh_s.distribute_dofs(fe_s);

	// set boundary conditions by formula
	// dirichlet b.c.
	FunctionParser<2> fp(4);
	fp.initialize("x,y", "y*(1-y)*4;0;0;0", {});
	typename FunctionMap<2>::type function_map;
	ZeroFunction<2> zero_bc(3);
	function_map[0] = &fp;
// 	function_map[1] = &zero_bc;
// 	function_map[3] = &zero_bc;
	FEValuesExtractors::Vector velocities(0);
	FEValuesExtractors::Scalar stress(3);
	VectorTools::interpolate_boundary_values(
			dh,
			function_map,
			constraints,
 			fe.component_mask(velocities)
			);
	// impermeability b.c.
	std::set<types::boundary_id> no_normal_flux_boundaries;
	no_normal_flux_boundaries.insert(slip_boundary_id);
	no_normal_flux_boundaries.insert(3);
	VectorTools::compute_no_normal_flux_constraints(
			dh,
			0,
			no_normal_flux_boundaries,
			constraints);
	constraints.close();

	bsp.reinit(3,3);
	
	
	BlockCompressedSparsityPattern c_sparsity(dh.block_info().global(), dh.block_info().global());
	DoFTools::make_sparsity_pattern (dh, c_sparsity, constraints, true);
	bsp.copy_from(c_sparsity);
	bsm.reinit(bsp);
	bv.reinit(dh.block_info().global());
	
	vp_mat = &bsm.block(0,2);
	sv_mat = &bsm.block(3,0);
 	v_sol.reinit(dh.block_info().global().block_size(0)*2);
	p_sol.reinit(dh.block_info().global().block_size(2));
	s_sol.reinit(dh.block_info().global().block_size(3));
	
	DoFTools::map_dof_to_boundary_indices(dh, {slip_boundary_id}, s_dof_map);
	std::vector<bool> s_dofs;
	DoFTools::extract_boundary_dofs(dh, fe.component_mask(stress), s_dofs, {slip_boundary_id});
	for (unsigned int i=0; i<s_dofs.size(); i++)
	  if (!s_dofs[i]) s_dof_map[i] = types::invalid_dof_index;
	
// 	std::vector<types::global_dof_index> v_local_dof_indices (fe_v.dofs_per_cell);
// 	std::vector<types::global_dof_index> p_local_dof_indices (fe_p.dofs_per_cell);
// 	std::vector<types::global_dof_index> s_local_dof_indices (fe_s.dofs_per_cell);
// 	vp_sparsity.reinit(dh_v.n_dofs(), dh_p.n_dofs(), 16);
// 	DoFHandler<2>::active_cell_iterator
// 	  v_cell = dh_v.begin_active(),
// 	  v_endc = dh_v.end(),
// 	  p_cell = dh_p.begin_active();
// 	for (; v_cell!=v_endc; ++v_cell, ++p_cell)
// 	{
// 		v_cell->get_dof_indices (v_local_dof_indices);
// 		p_cell->get_dof_indices (p_local_dof_indices);
// 
// 		for (unsigned int i=0; i<fe_v.dofs_per_cell; ++i)
// 		  for (unsigned int j=0; j<fe_p.dofs_per_cell; ++j)
// 		    vp_sparsity.add(v_local_dof_indices[i],p_local_dof_indices[j]);
// 	}
// 	vp_sparsity.compress();
// 	vp_mat.reinit(vp_sparsity);
// 	p_sol.reinit(dh_p.n_dofs());
// 
// 	v_sol.reinit (dh_v.n_dofs());
// 	v_rhs.reinit (dh_v.n_dofs());
// 
// 	// initialize stress solution and matrix
// 	DoFTools::map_dof_to_boundary_indices(dh_s, {slip_boundary_id}, s_dof_map);
// 	std::cout << "sv_sparsity dim: " << dh_s.n_boundary_dofs({slip_boundary_id}) << " " << dh_v.n_dofs() << std::endl;
// 	sv_sparsity.reinit(dh_s.n_boundary_dofs({slip_boundary_id}), dh_v.n_dofs(), 50);
// 	DoFHandler<2>::active_cell_iterator
// 	cell = dh_v.begin_active(),
// 	s_cell = dh_s.begin_active(),
// 	endc = dh_v.end();
// 	for (; cell!=endc; ++cell, ++s_cell)
// 	{
// 	  cell->get_dof_indices(v_local_dof_indices);
// 	  s_cell->get_dof_indices(s_local_dof_indices);
// 	  for (unsigned int face = 0; face < 4; ++face)
// 	  {
// 	    if (cell->at_boundary(face) && cell->face(face)->boundary_indicator() == slip_boundary_id)
// 	    {
// 	      for (unsigned int i=0; i<fe_v.dofs_per_cell; i++)
// 		for (unsigned int j=0; j<fe_s.dofs_per_cell; j++)
// 		  if (s_dof_map[s_local_dof_indices[j]] != types::invalid_dof_index)
// 		    sv_sparsity.add(s_dof_map[s_local_dof_indices[j]], v_local_dof_indices[i]);
// 	    }
// 	  }
// 	}
// 	s_sol.reinit(dh_s.n_boundary_dofs({slip_boundary_id}));
// 	sv_sparsity.compress();
// 	sv_mat.reinit(sv_sparsity);
	
	// set lower and upper bounds
	lb = new double[size()];
	ub = new double[size()];
	for (unsigned int i=0; i<p_sol.size(); i++)
	{
	  lb[i] = -1e12;
	  ub[i] = 1e12;
	}
	for (unsigned int i=0; i<s_sol.size(); i++)
	{
	   lb[p_sol.size()+i] = -sigma_bound;
	   ub[p_sol.size()+i] = sigma_bound;
	}
	
	std::cout << "Number of degrees of freedom (v/p/s): "
	  << v_sol.size() << "/"
	  << p_sol.size() << "/"
	  << s_sol.size()
	  << std::endl;
}


void StokesSlip::assemble_system ()
{
	QGauss<2>  quadrature_formula(2);
	FEValues<2> fv (fe, quadrature_formula,
			update_values | update_gradients | update_JxW_values);
	FEFaceValues<2> ffv(fe, quad1d, update_values | update_normal_vectors);
	FEValues<2> fv_v (fe_v, quadrature_formula,
			update_values | update_gradients | update_JxW_values | update_quadrature_points);
	FEValues<2> fv_p (fe_p, quadrature_formula,
			update_values | update_JxW_values);
	FEFaceValues<2> ffv_v(fe_v, quad1d, update_values | update_quadrature_points | update_JxW_values);
	FEFaceValues<2> ffv_s(fe_s, quad1d, update_values | update_normal_vectors);
	
	const unsigned int   dofs_per_cell = fe.dofs_per_cell;
	const unsigned int   v_dofs_per_cell = fe_v.dofs_per_cell;
	const unsigned int   p_dofs_per_cell = fe_p.dofs_per_cell;
	const unsigned int   s_dofs_per_cell = fe_s.dofs_per_cell;
	const unsigned int   n_q_points      = quadrature_formula.size();
// 	const unsigned int   n_stress_elems  = s_sol.size();

	FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
	FullMatrix<double>   v_cell_matrix (v_dofs_per_cell, v_dofs_per_cell);
	FullMatrix<double>   p_cell_matrix (v_dofs_per_cell, p_dofs_per_cell);
	Vector<double>       v_cell_rhs (v_dofs_per_cell);
	Vector<double>       cell_rhs (dofs_per_cell);

	std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
	std::vector<types::global_dof_index> v_local_dof_indices (v_dofs_per_cell);
	std::vector<types::global_dof_index> p_local_dof_indices (p_dofs_per_cell);
	std::vector<types::global_dof_index> s_local_dof_indices (s_dofs_per_cell);

	const FEValuesExtractors::Vector velocities (0);
	const FEValuesExtractors::Scalar pressure (2);
	const FEValuesExtractors::Scalar stress (3);

	std::vector<Tensor<2,2> > grad_phi_u (dofs_per_cell);
	std::vector<double>       div_phi_u  (dofs_per_cell);
	std::vector<double>       phi_p      (dofs_per_cell);
	std::vector<double>       phi_s      (dofs_per_cell);
	
	unsigned int i_stress = 0;
	DoFHandler<2>::active_cell_iterator
	  cell = dh.begin_active(),
	  v_cell = dh_v.begin_active(),
	  v_endc = dh_v.end(),
	  p_cell = dh_p.begin_active(),
	  s_cell = dh_s.begin_active();
	for (; v_cell!=v_endc; ++cell, ++v_cell, ++p_cell, ++s_cell)
	{
		fv.reinit(cell);
		fv_v.reinit(v_cell);
		fv_p.reinit(p_cell);

		cell_matrix = 0;
		cell_rhs = 0;
		v_cell_matrix = 0;
		p_cell_matrix = 0;
		v_cell_rhs = 0;
		
		cell->get_dof_indices (local_dof_indices);
		v_cell->get_dof_indices (v_local_dof_indices);
		p_cell->get_dof_indices (p_local_dof_indices);
		s_cell->get_dof_indices (s_local_dof_indices);

		for (unsigned int q=0; q<n_q_points; ++q)
		{
		  for (unsigned int k=0; k<dofs_per_cell; ++k)
		  {
		    grad_phi_u[k] = fv[velocities].symmetric_gradient (k, q);
		    div_phi_u[k]  = fv[velocities].divergence (k, q);
// 		  }
// 		  for (unsigned int k=0; k<p_dofs_per_cell; ++k)
		    phi_p[k] = fv[pressure].value (k, q);
		    phi_s[k] = fv[stress].value (k, q);
		  }

		  for (unsigned int i=0; i<dofs_per_cell; ++i)
		  {
// 		    v_cell_rhs(i) += fv_v[velocities].value(i, q)[0] * fv_v.JxW(q);
		    
		    for (unsigned int j=0; j<dofs_per_cell; ++j)
		    {
		      cell_matrix(i,j) += scalar_product(grad_phi_u[i], grad_phi_u[j]) * fv.JxW(q);
		  
// 		    for (unsigned int j=0; j<p_dofs_per_cell; ++j)
		      cell_matrix(i,j) -= div_phi_u[i] * phi_p[j] * fv.JxW(q);
		      cell_matrix(j,i) -= div_phi_u[i] * phi_p[j] * fv.JxW(q);
		    }
		  }
		}
		
		for (unsigned int face = 0; face < 4; ++face)
		{
		  if (v_cell->at_boundary(face) && v_cell->face(face)->boundary_indicator() == slip_boundary_id)
		  {
		    ffv_v.reinit(v_cell, face);
		    ffv_s.reinit(s_cell, face);
		    ffv.reinit(cell, face);
		    
		    for (unsigned int q=0; q<quad1d.size(); ++q)
		    {
		      Point<2> normal = ffv_s.normal_vector(q);
		      
		      for (unsigned int j=0; j<dofs_per_cell; ++j)
		      {
// 			unsigned int node_id = v_cell->face(face)->vertex_index(j);
// 			if (qp.distance(triangulation.get_vertices()[node_id]) > qp.distance(v_cell->face(face)->center())) continue;
			Point<2> tangent = {-normal[1], normal[0]};
			for (unsigned int i=0; i<dofs_per_cell; ++i)
			  if (s_dof_map[local_dof_indices[j]] != types::invalid_dof_index)
			  {
			    cell_matrix(i,j) -= ffv[velocities].value(i,q)*tangent*ffv[stress].value(j,q)*ffv.JxW(q);
			    cell_matrix(j,i) += ffv[velocities].value(i,q)*tangent*ffv[stress].value(j,q)*ffv.JxW(q);
// 			    sv_mat->add(s_dof_map[s_local_dof_indices[j]],v_local_dof_indices[i], (ffv_v[velocities].value(i, q)*tangent)*ffv_s.shape_value(j, q) * ffv_v.JxW(q)); // bazova funkce jen na polovine elementu!
			  }
		      }
		    }
		    i_stress++;
		  }
		}
		
// 		constraints.distribute_local_to_global (v_cell_matrix, v_cell_rhs,
// 							v_local_dof_indices,
// 							vv_mat, v_rhs);
		bsm.add(local_dof_indices, local_dof_indices, cell_matrix);
// 		v_rhs->add(v_local_dof_indices, v_cell_rhs);
		
// 		for (unsigned int i=0; i<v_dofs_per_cell; ++i)
// 		  for (unsigned int j=0; j<p_dofs_per_cell; ++j)
// 		    vp_mat->add (v_local_dof_indices[i], p_local_dof_indices[j], p_cell_matrix(i,j));
	}
	
	constraints.condense(bsm, bv);
	
	vv_sparsity.reinit(2,2);
	for (int i=0; i<2; i++)
	  for (int j=0; j<2; j++)
	    vv_sparsity.block(i,j).copy_from(bsp.block(i,j));
	vv_sparsity.collect_sizes();
	vv_mat.reinit(vv_sparsity);
	for (int i=0; i<2; i++)
	  for (int j=0; j<2; j++)
	  {
	    vv_mat.block(i,j).reinit(bsp.block(i,j));
	    vv_mat.block(i,j).copy_from(bsm.block(i,j));
	  }
	v_rhs.reinit(2, dh.block_info().global().block_size(0));
	v_rhs.block(0) = bv.block(0);
	v_rhs.block(1) = bv.block(1);
	
// 	std::cout << "vv_mat:\n";
// 	vv_mat.print(std::cout);
// 	
// 	std::cout << "\n\nv_rhs:\n";
// 	v_rhs.print(std::cout);
}






void StokesSlip::solve (Vector<double> &rhs, bool distribute)
{
	SparseDirectUMFPACK solver;
	
	solver.solve(vv_mat, rhs);
// 	if (distribute) constraints.distribute(rhs);
}





void StokesSlip::output_results(double *x)
{
  
  for (unsigned int i=0; i<p_sol.size(); i++) p_sol(i) = x[i];
  for (unsigned int i=0; i<s_sol.size(); i++) s_sol(i) = x[p_sol.size()+i];
  
  // u = A^-1(f + C^T*s - B*p)
  // v_sol = vv_mat^-1(v_rhs + sv_mat^T*s_sol - vp_mat*p_sol)
  v_sol = v_rhs;
  sv_mat->Tvmult_add(v_sol, s_sol);
  Vector<double> w(v_sol.size());
  vp_mat->vmult(w, p_sol);
  v_sol.add(-1, w);
  solve(v_sol, true);
  
  Vector<double> s_global_sol(dh_s.n_dofs());
  for (unsigned int i=0; i<s_global_sol.size(); i++)
    s_global_sol(i) = (s_dof_map[i] == types::invalid_dof_index)? 0 : s_sol(s_dof_map[i]);
  
  std::vector<std::string> v_names(2, "velocity");
  std::vector<std::string> p_names(1, "pressure");
  std::vector<std::string> s_names(1, "shear_stress");

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    v_component_interpretation(2, DataComponentInterpretation::component_is_part_of_vector),
    p_component_interpretation(1, DataComponentInterpretation::component_is_scalar),
    s_component_interpretation(1, DataComponentInterpretation::component_is_scalar);

  DataOut<2> data_out;
  data_out.attach_triangulation(triangulation);
  data_out.add_data_vector(dh_v, v_sol, v_names, v_component_interpretation);
  data_out.add_data_vector(dh_p, p_sol, p_names, p_component_interpretation);
  data_out.add_data_vector(dh_s, s_global_sol, s_names, s_component_interpretation);
  
  data_out.build_patches ();

  std::ofstream output ("solution.vtk");
  data_out.write_vtk (output);

  std::ofstream f("shear_stress.dat");
  for (unsigned int i=0; i<s_sol.size(); i++)
    f << x[p_sol.size()+i] << std::endl;
  f.close();
}





double StokesSlip::evaluate_fg(const double *x, double *grad)
{
  for (unsigned int i=0; i<p_sol.size(); i++) p_sol(i) = x[i];
  for (unsigned int i=0; i<s_sol.size(); i++) s_sol(i) = x[p_sol.size()+i];
  
  
  // z = A^-1*f
  Vector<double> z;
  z = v_rhs;
  solve(z, false);
  
  // Bp = B*p
  Vector<double> Bp(v_sol.size());
  vp_mat->vmult(Bp,p_sol);
  // w = A^-1(C^T*s-B*p)
  Vector<double> w(Bp);
  w *= -1;
  sv_mat->Tvmult_add(w,s_sol);
  solve(w, false);
  
  // zphw = z + 0.5*w
  Vector<double> zphw(z);
  zphw.add(0.5, w);
  // Czphw = C*zphw
  Vector<double> Czphw(s_sol.size());
  sv_mat->vmult(Czphw, zphw);
  
  Vector<double> zpw(z);
  zpw.add(w);
  Vector<double> Btzpw(p_sol.size());
  vp_mat->Tvmult(Btzpw, zpw);
  
  Vector<double> Czpw(s_sol.size());
  sv_mat->vmult(Czpw, zpw);
  
  for (unsigned int i=0; i<p_sol.size(); i++) grad[i] = -Btzpw(i);
  for (unsigned int i=0; i<s_sol.size(); i++) grad[p_sol.size()+i] = Czpw(i);
  
  // s.C(z+0.5w) - B*p.(z+0.5w)
  return (s_sol*Czphw) - (Bp*zphw);
}


double *StokesSlip::lower_bounds()
{
  return lb;
}


double *StokesSlip::upper_bounds()
{
  return ub;
}




double myfunc(unsigned n, const double *x, double *grad, void *my_func_data)
{
  double val = stokes_problem.evaluate_fg(x, grad);
  return val;
}


int main ()
{
	nlopt_opt opt = nlopt_create(NLOPT_LD_MMA, stokes_problem.size()); /* algorithm and dimensionality */
	nlopt_set_lower_bounds(opt, stokes_problem.lower_bounds());
	nlopt_set_upper_bounds(opt, stokes_problem.upper_bounds());
	nlopt_set_min_objective(opt, myfunc, NULL);
	nlopt_set_xtol_rel(opt, 1e-5);
	
	double *x;
	double minf; /* the minimum objective value, upon return */
	
	x = new double[stokes_problem.size()];
	for (unsigned int i=0; i<stokes_problem.size(); i++) x[i] = 0;

	if (nlopt_optimize(opt, x, &minf) < 0) {
	    printf("nlopt failed!\n");
	}
	else {
	    printf("found minimum = %0.10g\n", minf);
	    
	    stokes_problem.output_results(x);
	}
	
	nlopt_destroy(opt);
	delete[] x;

	return 0;
}
