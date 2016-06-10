#include "parameters.hh"
#include <sys/stat.h>
#include <iostream>

using namespace dealii;
using namespace Parameters;



std::vector<std::string> inline split_string(const std::string &source, const char *delimiter = " ", bool keepEmpty = false)
{
    std::vector<std::string> results;

    size_t prev = 0;
    size_t next = 0;

    while ((next = source.find_first_of(delimiter, prev)) != std::string::npos)
    {
        if (keepEmpty || (next - prev != 0))
            results.push_back(source.substr(prev, next - prev));
        prev = next + 1;
    }

    if (prev < source.size())
        results.push_back(source.substr(prev));

    return results;
}

std::map<int,std::string> string_to_map_int_string(const std::string &str, const char *delim, const char *delim_key_value)
{
    std::map<int,std::string> result;
    std::vector<std::string> v = split_string(str, delim);

    for (std::string const & s : v) {
        auto i = s.find(delim_key_value);
        result[atoi(s.substr(0,i).c_str())] = s.substr(i + 1);
    }

    return result;
}



AllParameters::AllParameters(const std::string &input_file)
{
	struct stat buffer;
	if (stat(input_file.c_str(), &buffer) != 0)
	{
		std::cerr << "Cannot open input file '" << input_file << "'!" << std::endl;
		exit(1);
	}
	declare_parameters(prm);
	prm.read_input(input_file);
	parse_parameters(prm);
}

void AllParameters::declare_parameters(ParameterHandler &prm)
{
// 	Materials::declare_parameters(prm);
	BoundaryConditions::declare_parameters(prm);
	BulkParameters::declare_parameters(prm);
    CostFunction::declare_parameters(prm);
    Numerics::declare_parameters(prm);
	IO::declare_parameters(prm);
    ShapeOptimization::declare_parameters(prm);
}

void AllParameters::parse_parameters(ParameterHandler &prm)
{
// 	Materials::parse_parameters(prm);
	BoundaryConditions::parse_parameters(prm);
	BulkParameters::parse_parameters(prm);
    CostFunction::parse_parameters(prm);
    Numerics::parse_parameters(prm);
	IO::parse_parameters(prm);
    ShapeOptimization::parse_parameters(prm);
}

void AllParameters::print(std::ostream& out)
{
  prm.print_parameters(out, ParameterHandler::Description);
}





// void Materials::declare_parameters(ParameterHandler &prm)
// {
// 	prm.enter_subsection("Materials");
// 		prm.enter_subsection("Matrix");
// 			prm.declare_entry("Young modulus", "0.9e9", Patterns::Double(), "Young modulus of matrix");
// 			prm.declare_entry("Poisson ratio", "0.4",   Patterns::Double(), "Poisson's ratio of matrix");
// 		prm.leave_subsection();
// 
// 		prm.enter_subsection("Fiber");
// 			prm.declare_entry("Young modulus", "19.1e9",   Patterns::Double(), "Young modulus of fiber");
// 			prm.declare_entry("Poisson ratio", "0.4",      Patterns::Double(), "Poisson's ratio of fiber");
// 			prm.declare_entry("Fiber volume ratio", "0.5", Patterns::Double(), "Volume ratio of fiber in reinforcement");
// 			prm.declare_entry("Material id", "-1", Patterns::Integer(), "Id of reinforcement material in mesh");
// 			prm.declare_entry("Use 1d mesh", "false", Patterns::Bool(), "If true, fibers are read from separate mesh file as 1D elements.");
// 		prm.leave_subsection();
// 	prm.leave_subsection();
// }
// 
// void Materials::parse_parameters(ParameterHandler &prm)
// {
// 	prm.enter_subsection("Materials");
// 		prm.enter_subsection("Matrix");
// 			Young_modulus_matrix = prm.get_double("Young modulus");
// 			Poisson_ratio_matrix = prm.get_double("Poisson ratio");
// 		prm.leave_subsection();
// 
// 		prm.enter_subsection("Fiber");
// 			Young_modulus_fiber = prm.get_double("Young modulus");
// 			Poisson_ratio_fiber = prm.get_double("Poisson ratio");
// 			Fiber_volume_ratio  = prm.get_double("Fiber volume ratio");
// 			Reinforcement_material_id = prm.get_integer("Material id");
// 			use_1d_fibers = prm.get_bool("Use 1d mesh");
// 		prm.leave_subsection();
// 	prm.leave_subsection();
// }

void BoundaryConditions::declare_parameters(ParameterHandler &prm)
{
	prm.enter_subsection("BoundaryConditions");
		prm.declare_entry("Velocity",    "", Patterns::Map(Patterns::Integer(), Patterns::Anything()));
        prm.declare_entry("Velocity tangent",    "", Patterns::Map(Patterns::Integer(), Patterns::Anything()));
        prm.declare_entry("Velocity normal",     "", Patterns::Map(Patterns::Integer(), Patterns::Anything()));
        prm.declare_entry("Velocity slip",       "", Patterns::Map(Patterns::Integer(), Patterns::Anything()));
		prm.declare_entry("Traction",            "", Patterns::Map(Patterns::Integer(), Patterns::Anything()));
	prm.leave_subsection();
}

void BoundaryConditions::parse_parameters(ParameterHandler &prm)
{
	prm.enter_subsection("BoundaryConditions");
		velocity         = string_to_map_int_string(prm.get("Velocity"),         ",", ":");
        velocity_tangent = string_to_map_int_string(prm.get("Velocity tangent"), ",", ":");
        velocity_normal  = string_to_map_int_string(prm.get("Velocity normal"),  ",", ":");
        velocity_slip    = string_to_map_int_string(prm.get("Velocity slip"),    ",", ":");
		traction         = string_to_map_int_string(prm.get("Traction"),        ",", ":");
	prm.leave_subsection();
}

void BulkParameters::declare_parameters(ParameterHandler &prm)
{
	prm.enter_subsection("BulkParameters");
		prm.declare_entry("Force", "", Patterns::Map(Patterns::Integer(), Patterns::Anything()));
		prm.declare_entry("Symmetric gradient", "true", Patterns::Bool());
	prm.leave_subsection();
}

void BulkParameters::parse_parameters(ParameterHandler &prm)
{
	prm.enter_subsection("BulkParameters");
		force = string_to_map_int_string(prm.get("Force"), ",", ":");
		sym_grad = prm.get_bool("Symmetric gradient");
	prm.leave_subsection();
}


void CostFunction::declare_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("CostFunction");
        prm.declare_entry("Boundary integral", "", Patterns::Map(Patterns::Integer(), Patterns::Anything(), 0, 1000, "@"));
        prm.declare_entry("Volume integral",   "", Patterns::Map(Patterns::Integer(), Patterns::Anything(), 0, 1000, "@"));
    prm.leave_subsection();
}

void CostFunction::parse_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("CostFunction");
        boundary_integral = string_to_map_int_string(prm.get("Boundary integral"), "@", ":");
        volume_integral   = string_to_map_int_string(prm.get("Volume integral"),   "@", ":");
    prm.leave_subsection();
}


void ShapeOptimization::declare_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("ShapeOptimization");
        prm.declare_entry("Left point",  "0;0;0", Patterns::Anything());
        prm.declare_entry("Right point", "1;0;0", Patterns::Anything());
        prm.declare_entry("Top point", "0;1;0", Patterns::Anything());
//         prm.declare_entry("Box height",  "1", Patterns::Double());
        prm.declare_entry("N_control_pts", "8", Patterns::Integer());
        prm.declare_entry("Max_it", "0", Patterns::Integer());
        prm.declare_entry("Initial shape", "", Patterns::List(Patterns::Double()), "Initial control points of bezier curve describing shape.");
        prm.declare_entry("F_max", "", Patterns::Anything());
        prm.declare_entry("F_min", "", Patterns::Anything());
        prm.declare_entry("G_max", "1e308", Patterns::Double());
        prm.declare_entry("H_max", "1e308", Patterns::Double());
        prm.declare_entry("Log file", "", Patterns::FileName(), "Log file for optimization");
    prm.leave_subsection();
}

void ShapeOptimization::parse_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("ShapeOptimization");
        std::vector<std::string> pstr = split_string(prm.get("Left point"), ";");
        p1 = { atof(pstr[0].c_str()), atof(pstr[1].c_str()), atof(pstr[2].c_str()) };
        pstr = split_string(prm.get("Right point"), ";");
        p2 = { atof(pstr[0].c_str()), atof(pstr[1].c_str()), atof(pstr[2].c_str()) };
        pstr = split_string(prm.get("Top point"), ";");
        ptop = { atof(pstr[0].c_str()), atof(pstr[1].c_str()), atof(pstr[2].c_str()) };
        np = prm.get_integer("N_control_pts");
        maxit = prm.get_integer("Max_it");
        std::vector<std::string> init_str = split_string(prm.get("Initial shape"), ",");
        if (init_str.size() == np)
        {
          for (auto s : init_str) init_guess.push_back(atof(s.c_str()));
        }
        else
        {
          init_guess.resize(np, 0.);
        }
        f_max = prm.get("F_max");
        f_min = prm.get("F_min");
        g_max = prm.get_double("G_max");
        h_max = prm.get_double("H_max");
        logfile = prm.get("Log file");
    prm.leave_subsection();
}


void Numerics::declare_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Numerics");
        prm.declare_entry("Epsilon",  "1e-6", Patterns::Double(),  "Regularization parameter in slip b.c.");
        prm.declare_entry("Abs_tol",  "1e-6", Patterns::Double(),  "Absolute tolerance in nonlinear solver");
        prm.declare_entry("Rel_tol",  "1e-6", Patterns::Double(),  "Relative tolerance in nonlinear solver");
        prm.declare_entry("Max_iter", "10", Patterns::Integer(), "Maximal number of iterations in nonlinear solver");
    prm.leave_subsection();
}

void Numerics::parse_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Numerics");
        epsilon  = prm.get_double ("Epsilon");
        a_tol    = prm.get_double ("Abs_tol");
        r_tol    = prm.get_double ("Rel_tol");
        max_iter = prm.get_integer("Max_iter");
    prm.leave_subsection();
}



void IO::declare_parameters(ParameterHandler &prm)
{
	prm.enter_subsection("InputOutput");
        prm.declare_entry("Dimension", "2", Patterns::Integer(), "Space dimension of the problem.");
 		prm.declare_entry("Mesh", "", Patterns::FileName(), "Mesh file name");
		prm.declare_entry("Output", "", Patterns::FileName(), "Output file name base");
	prm.leave_subsection();
}

void IO::parse_parameters(ParameterHandler &prm)
{
	prm.enter_subsection("InputOutput");
        dim = prm.get_integer("Dimension");
		mesh_file = prm.get("Mesh");
		output_file_base = prm.get("Output");
	prm.leave_subsection();
}



