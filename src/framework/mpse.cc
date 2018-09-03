//--------------------------------------------------------------------------
// Copyright (C) 2014-2017 Cisco and/or its affiliates. All rights reserved.
//
// This program is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License Version 2 as published
// by the Free Software Foundation.  You may not use, modify or distribute
// this program under any other version of the GNU General Public License.
//
// This program is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// General Public License for more details.
//
// You should have received a copy of the GNU General Public License along
// with this program; if not, write to the Free Software Foundation, Inc.,
// 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
//--------------------------------------------------------------------------
// mpse.cc author Russ Combs <rucombs@cisco.com>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <string.h>

#include "mpse.h"

#include "profiler/profiler_defs.h"

using namespace std;

// this is accumulated only for fast pattern
// searches for the detection engine
static THREAD_LOCAL uint64_t s_bcnt=0;

THREAD_LOCAL ProfileStats mpsePerfStats;

//-------------------------------------------------------------------------
// base stuff
//-------------------------------------------------------------------------

Mpse::Mpse(const char* m, bool use_gc)
{
    method = m;
    inc_global_counter = use_gc;
    verbose = 0;

    buffer = (uint8_t*) malloc ( MPSE_BUFFER_SIZE * sizeof( char ) );
    loop_count = 0;
    offset = 0;
}

int Mpse::search(
    const unsigned char* T, int n, MpseMatch match,
    void* context, int* current_state)
{
    Profile profile(mpsePerfStats);

    int ret = _search(T, n, match, context, current_state);

    if ( inc_global_counter )
        s_bcnt += n;

    return ret;
}

int Mpse::search_all(
    const unsigned char* T, int n, MpseMatch match,
    void* context, int* current_state)
{
    loop_count++;
    memcpy( buffer + offset, T, n * sizeof( char ) );
    offset += n;

    if (loop_count < 150) 
    {
        return 0;
    }

    int ret = _search(buffer, offset, match, context, current_state);
    loop_count = 0;
    offset = 0;

    return ret;
}

Mpse::~Mpse()
{
    free(buffer);
}

uint64_t Mpse::get_pattern_byte_count()
{
    return s_bcnt;
}

void Mpse::reset_pattern_byte_count()
{
    s_bcnt = 0;
}

