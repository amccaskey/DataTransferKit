//---------------------------------------------------------------------------//
/*
 Copyright (c) 2012, Stuart R. Slattery
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are
 met:

 *: Redistributions of source code must retain the above copyright
 notice, this list of conditions and the following disclaimer.

 *: Redistributions in binary form must reproduce the above copyright
 notice, this list of conditions and the following disclaimer in the
 documentation and/or other materials provided with the distribution.

 *: Neither the name of the University of Wisconsin - Madison nor the
 names of its contributors may be used to endorse or promote products
 derived from this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
//---------------------------------------------------------------------------//
/*!
 * \file   tstUserApplication.cpp
 * \author Stuart Slattery
 * \brief  UserApplication unit tests.
 */
//---------------------------------------------------------------------------//
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE LibmeshUserAppTester

#include <boost/test/included/unit_test.hpp>

#include <DTK_UserApplication.hpp>
#include <DTK_UserDataInterface.hpp>
#include <DTK_UserFunctionRegistry.hpp>
#include <DTK_View.hpp>

#include <libmesh/dof_map.h>
#include <libmesh/elem.h>
#include <libmesh/equation_systems.h>
#include <libmesh/exodusII_io.h>
#include <libmesh/explicit_system.h>
#include <libmesh/fe.h>
#include <libmesh/libmesh.h>
#include <libmesh/mesh.h>
#include <libmesh/node.h>
#include <libmesh/numeric_vector.h>
#include <libmesh/parallel.h>
#include <libmesh/quadrature_gauss.h>
#include <libmesh/system.h>
#include <libmesh/mesh_generation.h>

#include <Kokkos_Core.hpp>

#include <Teuchos_Array.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_DefaultMpiComm.hpp>
#include <Teuchos_FancyOStream.hpp>
#include <Teuchos_OpaqueWrapper.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_Tuple.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_VerboseObject.hpp>
#include <Teuchos_XMLParameterListCoreHelpers.hpp>
#include <Teuchos_ScalarTraits.hpp>
#include <Teuchos_UnitTestHarness.hpp>

#include <memory>

namespace LibmeshAppTest {

//---------------------------------------------------------------------------//
// User functions.
//---------------------------------------------------------------------------//
// Get the size parameters for building a node list.
void nodeListSize(std::shared_ptr<void> user_data, unsigned &space_dim,
		size_t &local_num_nodes, bool &has_ghosts) {
	auto u = std::static_pointer_cast<libMesh::Mesh>(user_data);
	space_dim = u->mesh_dimension();
	local_num_nodes = u->n_local_nodes();
	has_ghosts = false;//u->ghost_elements_begin() != u->ghost_elements_end();
}

//---------------------------------------------------------------------------//
// Get the data for a node list.
void nodeListData(std::shared_ptr<void> user_data,
		DataTransferKit::View<DataTransferKit::Coordinate> coordinates,
		DataTransferKit::View<bool> is_ghost_node) {
	auto u = std::static_pointer_cast<libMesh::Mesh>(user_data);
	auto dim = u->mesh_dimension();
	auto num_nodes = u->n_local_nodes();
	int counter = 0;
	libMesh::MeshBase::node_iterator nodes_begin = u->local_nodes_begin();
	libMesh::MeshBase::node_iterator nodes_end = u->local_nodes_end();
	for (auto node_it = nodes_begin; node_it != nodes_end; ++node_it) {
		for (unsigned d = 0; d < dim; ++d) {
			coordinates[num_nodes * d + counter] = (*node_it)->operator()(d);
		}
		counter++;
	}
}

//---------------------------------------------------------------------------//
// Get the size parameters for building a bounding volume list.
template <class Scalar, class ExecutionSpace>
void boundingVolumeListSize( std::shared_ptr<void> user_data,
                             unsigned &space_dim, size_t &local_num_volumes,
                             bool &has_ghosts )
{
    auto u = std::static_pointer_cast<libMesh::Mesh>(
        user_data );

    space_dim = u->mesh_dimension();
//    local_num_volumes = u->_size_1;
    has_ghosts = false;
}

//---------------------------------------------------------------------------//

}

struct TestFixture {
	static TestFixture*& instance() {static TestFixture* inst = 0; return inst;}
	TestFixture() {
		BOOST_TEST_MESSAGE("setup fixture");
		instance() = this;
		const std::string argv_string = "unit_test";
		const char *argv_char = argv_string.c_str();
		 Teuchos::GlobalMPISession mpiSession(&boost::unit_test::framework::master_test_suite().argc,
				 &boost::unit_test::framework::master_test_suite().argv);
		auto comm = Teuchos::DefaultComm<int>::getComm();
		auto mpi_comm = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(
				comm);
		auto opaque_comm = mpi_comm->getRawMpiComm();
		auto raw_comm = (*opaque_comm)();

		// Create the mesh.
		BOOST_VERIFY(!libMesh::initialized());
		libMesh::LibMeshInit libmesh_init(boost::unit_test::framework::master_test_suite().argc,
				 boost::unit_test::framework::master_test_suite().argv, raw_comm);
		BOOST_VERIFY(libMesh::initialized());
		BOOST_VERIFY((int)libmesh_init.comm().rank() == comm->getRank());
		mesh = std::make_shared<libMesh::Mesh>(libmesh_init.comm());
		libMesh::MeshTools::Generation::build_cube(*mesh.get(), 4, 4, 4, 0.0, 1.0,
				0.0, 1.0, 0.0, 1.0, libMesh::HEX8);

		// Set the user functions.
		registry = std::make_shared<
				DataTransferKit::UserFunctionRegistry<double>>();
		registry->setNodeListSizeFunction(LibmeshAppTest::nodeListSize, mesh);
		registry->setNodeListDataFunction(LibmeshAppTest::nodeListData, mesh);

		BOOST_VERIFY(mesh->n_local_nodes() == 125);
		BOOST_VERIFY(mesh);
	}
	~TestFixture() {
		BOOST_TEST_MESSAGE("teardown fixture");
	}

	std::shared_ptr<libMesh::Mesh> mesh;
	std::shared_ptr<
			DataTransferKit::UserFunctionRegistry<double>> registry;

};

BOOST_GLOBAL_FIXTURE(TestFixture);
//____________________________________________________________________________//

//BOOST_FIXTURE_TEST_SUITE( s, TestFixture )

BOOST_AUTO_TEST_CASE(checkNodeList) {
	auto fixture = TestFixture::instance();
	auto mesh = fixture->mesh;
	// Get a node list.
	// Create the user application.
	auto user_app = std::make_shared<
			DataTransferKit::UserApplication<double,
					Kokkos::Serial::execution_space>>(fixture->registry);

	auto dim = mesh->mesh_dimension();
	auto node_list = user_app->getNodeList();
	auto nNodes = mesh->n_local_nodes();
	BOOST_VERIFY(nNodes == 125);
	BOOST_VERIFY(node_list.coordinates.size() == 375);
	std::vector<double> expectedCoords { 0, 0, 0, 0.25, 0, 0, 0.25, 0.25, 0, 0,
			0.25, 0, 0, 0, 0.25, 0.25, 0, 0.25, 0.25, 0.25, 0.25, 0, 0.25, 0.25,
			0.5, 0, 0, 0.5, 0.25, 0, 0.5, 0, 0.25, 0.5, 0.25, 0.25, 0.75, 0, 0,
			0.75, 0.25, 0, 0.75, 0, 0.25, 0.75, 0.25, 0.25, 1, 0, 0, 1, 0.25, 0,
			1, 0, 0.25, 1, 0.25, 0.25, 0.25, 0.5, 0, 0, 0.5, 0, 0.25, 0.5, 0.25,
			0, 0.5, 0.25, 0.5, 0.5, 0, 0.5, 0.5, 0.25, 0.75, 0.5, 0, 0.75, 0.5,
			0.25, 1, 0.5, 0, 1, 0.5, 0.25, 0.25, 0.75, 0, 0, 0.75, 0, 0.25,
			0.75, 0.25, 0, 0.75, 0.25, 0.5, 0.75, 0, 0.5, 0.75, 0.25, 0.75,
			0.75, 0, 0.75, 0.75, 0.25, 1, 0.75, 0, 1, 0.75, 0.25, 0.25, 1, 0, 0,
			1, 0, 0.25, 1, 0.25, 0, 1, 0.25, 0.5, 1, 0, 0.5, 1, 0.25, 0.75, 1,
			0, 0.75, 1, 0.25, 1, 1, 0, 1, 1, 0.25, 0, 0, 0.5, 0.25, 0, 0.5,
			0.25, 0.25, 0.5, 0, 0.25, 0.5, 0.5, 0, 0.5, 0.5, 0.25, 0.5, 0.75, 0,
			0.5, 0.75, 0.25, 0.5, 1, 0, 0.5, 1, 0.25, 0.5, 0.25, 0.5, 0.5, 0,
			0.5, 0.5, 0.5, 0.5, 0.5, 0.75, 0.5, 0.5, 1, 0.5, 0.5, 0.25, 0.75,
			0.5, 0, 0.75, 0.5, 0.5, 0.75, 0.5, 0.75, 0.75, 0.5, 1, 0.75, 0.5,
			0.25, 1, 0.5, 0, 1, 0.5, 0.5, 1, 0.5, 0.75, 1, 0.5, 1, 1, 0.5, 0, 0,
			0.75, 0.25, 0, 0.75, 0.25, 0.25, 0.75, 0, 0.25, 0.75, 0.5, 0, 0.75,
			0.5, 0.25, 0.75, 0.75, 0, 0.75, 0.75, 0.25, 0.75, 1, 0, 0.75, 1,
			0.25, 0.75, 0.25, 0.5, 0.75, 0, 0.5, 0.75, 0.5, 0.5, 0.75, 0.75,
			0.5, 0.75, 1, 0.5, 0.75, 0.25, 0.75, 0.75, 0, 0.75, 0.75, 0.5, 0.75,
			0.75, 0.75, 0.75, 0.75, 1, 0.75, 0.75, 0.25, 1, 0.75, 0, 1, 0.75,
			0.5, 1, 0.75, 0.75, 1, 0.75, 1, 1, 0.75, 0, 0, 1, 0.25, 0, 1, 0.25,
			0.25, 1, 0, 0.25, 1, 0.5, 0, 1, 0.5, 0.25, 1, 0.75, 0, 1, 0.75,
			0.25, 1, 1, 0, 1, 1, 0.25, 1, 0.25, 0.5, 1, 0, 0.5, 1, 0.5, 0.5, 1,
			0.75, 0.5, 1, 1, 0.5, 1, 0.25, 0.75, 1, 0, 0.75, 1, 0.5, 0.75, 1,
			0.75, 0.75, 1, 1, 0.75, 1, 0.25, 1, 1, 0, 1, 1, 0.5, 1, 1, 0.75, 1,
			1, 1, 1, 1 };

	unsigned counter = 0;
	for (unsigned i = 0; i < nNodes; ++i) {
		for (unsigned d = 0; d < dim; ++d) {
			BOOST_VERIFY(node_list.coordinates(i, d) == expectedCoords[counter]);
			counter++;
		}
	}
}

//BOOST_AUTO_TEST_SUITE_END()
