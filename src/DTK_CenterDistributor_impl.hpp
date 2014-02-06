//---------------------------------------------------------------------------//
/*
  Copyright (c) 2014, Stuart R. Slattery
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  *: Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  *: Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  *: Neither the name of the Oak Ridge National Laboratory nor the
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
 * \file   DTK_CenterDistributor_impl.hpp
 * \author Stuart R. Slattery
 * \brief  Global acenter distributor.
 */
//---------------------------------------------------------------------------//

#ifndef DTK_CENTERDISTRIBUTOR_IMPL_HPP
#define DTK_CENTERDISTRIBUTOR_IMPL_HPP

#include <algorithm>
#include <limits>

#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_Array.hpp>

namespace DataTransferKit
{
//---------------------------------------------------------------------------//
/*!
 * \brief Constructor.
 */
template<int DIM>
CenterDistributor<DIM>::CenterDistributor(
	const Teuchos::RCP<const Teuchos::Comm<int> >& comm,
	const Teuchos::ArrayView<const double>& source_centers,
	const Teuchos::ArrayView<const double>& target_centers,
	const double radius,
	Teuchos::Array<double>& target_decomp_source_centers )
    : d_distributor( new Tpetra::Distributor(comm) )
{
    DTK_REQUIRE( 0 == source_centers.size() % DIM );
    DTK_REQUIRE( 0 == target_centers.size() % DIM );

    // Build the import/export data.
    Teuchos::Array<int> export_procs;
    {
	// Compute the radius to expand the local domain with.
	double radius_tol = 1.0e-2;
	double radius_expand = radius * (1.0 + radius_tol);

	// Gather the bounding domains for each proc.
	CloudDomain<DIM> local_domain = localCloudDomain( target_centers );
	local_domain.expand( radius_expand );
	Teuchos::Array<CloudDomain<DIM> > global_domains( comm->getSize() );
	Teuchos::gatherAll<int,CloudDomain<DIM> >( *comm,
						   1,
						   &local_domain,
						   global_domains.size(),
						   global_domains.getRawPtr() );

	// Find the procs to which the sources will be sent.
	Teuchos::ArrayView<const double> source_point;
	for ( unsigned source_id = 0; 
	      source_id < source_centers.size() / DIM;
	      ++source_id )
	{
	    source_point = source_centers.view( DIM*source_id, DIM );
	    for ( unsigned b = 0; b < global_domains.size(); ++b )
	    {
		if ( global_domains[b].pointInDomain(source_point) )
		{
		    export_procs.push_back( b );
		    d_export_ids.push_back( source_id );
		}
	    }
	}
    }
    DTK_CHECK( d_export_ids.size() == export_procs.size() );
    d_num_exports = d_export_ids.size();

    // Create the communication plan.
    Teuchos::ArrayView<int> export_procs_view = export_procs();
    d_num_imports = d_distributor->createFromSends( export_procs_view );
    export_procs.clear();

    // Move the source center coordinates to the target decomposition.
    target_decomp_source_centers.resize( d_num_imports * DIM );
    Teuchos::Array<unsigned>::const_iterator export_id_it;
    Teuchos::Array<double> src_dim_coords( d_num_exports );
    Teuchos::Array<double>::iterator src_dim_it;
    Teuchos::Array<double> tgt_dim_coords( d_num_imports );
    Teuchos::Array<double>::iterator tgt_dim_it;
    Teuchos::ArrayView<const double> src_dim_view = src_dim_coords();
    Teuchos::ArrayView<double> tgt_dim_view = tgt_dim_coords();
    Teuchos::Array<double>::iterator src_tgt_decomp_it;
    for ( int d = 0; d < DIM; ++d )
    {
	// Unroll the coordinates to handle cases where single source centers
	// may have multiple destinations.
	for ( export_id_it = d_export_ids.begin(), 
		src_dim_it = src_dim_coords.begin();
	      export_id_it != d_export_ids.end();
	      ++export_id_it, ++src_dim_it )
	{
	    *src_dim_it = source_centers[ (*export_id_it)*DIM + d ];
	}

	// Distribute.
	d_distributor->doPostsAndWaits( src_dim_view, 1, tgt_dim_view );

	// Extract the coordinates into the local array.
	for ( src_tgt_decomp_it = target_decomp_source_centers.begin(),
		     tgt_dim_it = tgt_dim_coords.begin();
	      src_tgt_decomp_it != target_decomp_source_centers.end();
	      ++tgt_dim_it )
	{
	    *(src_tgt_decomp_it+d) = *tgt_dim_it;
	    src_tgt_decomp_it += DIM;
	}
    }
}

//---------------------------------------------------------------------------//
/*!
 * \brief Given a set of scalar values at the given source centers in the
 * source decomposition, distribute them to the target decomposition.
 */
template<int DIM>
template<class T>
void CenterDistributor<DIM>::distribute( 
    const Teuchos::ArrayView<const T>& source_decomp_data,
    const Teuchos::ArrayView<T>& target_decomp_data ) const
{
    DTK_REQUIRE( d_num_imports == target_decomp_data.size() );

    // Unroll the source data to handle cases where single data points may
    // have multiple destinations.
    Teuchos::Array<unsigned>::const_iterator export_id_it;
    Teuchos::Array<T> src_data( d_num_exports );
    typename Teuchos::Array<T>::iterator src_it;
    for ( export_id_it = d_export_ids.begin(), 
		src_it = src_data.begin();
	  export_id_it != d_export_ids.end();
	  ++export_id_it, ++src_it )
    {
	DTK_CHECK( *export_id_it < source_decomp_data.size() );
	*src_it = source_decomp_data[ *export_id_it ];
    }

    // Distribute.
    Teuchos::ArrayView<const T> src_view = src_data();
    d_distributor->doPostsAndWaits( src_view, 1, target_decomp_data );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Compute the bounding domain of the local set of centers.
 */
template<int DIM>
CloudDomain<DIM> CenterDistributor<DIM>::localCloudDomain(
    const Teuchos::ArrayView<const double>& centers ) const
{
    Teuchos::Array<double> bounds( 2*DIM, 0.0 );

    if ( centers.size() > 0 )
    {
	for ( int d = 0; d < DIM; ++d )
	{
	    bounds[2*d] = std::numeric_limits<double>::max();
	    bounds[2*d+1] = std::numeric_limits<double>::min();
	}
    }

    Teuchos::ArrayView<const double>::const_iterator center_it;
    for ( center_it = centers.begin(); center_it != centers.end(); )
    {
	for ( int d = 0; d < DIM; ++d )
	{
	    bounds[2*d] = std::min( bounds[2*d], *(center_it+d) );
	    bounds[2*d+1] = std::max( bounds[2*d+1], *(center_it+d) );
	}
	center_it += DIM;
    }

    return CloudDomain<DIM>( bounds.getRawPtr() );
}

//---------------------------------------------------------------------------//

} // end namespace DataTransferKit

//---------------------------------------------------------------------------//

#endif // end DTK_CENTERDISTRIBUTOR_IMPL_HPP

//---------------------------------------------------------------------------//
// end DTK_CenterDistributor_impl.hpp
//---------------------------------------------------------------------------//

