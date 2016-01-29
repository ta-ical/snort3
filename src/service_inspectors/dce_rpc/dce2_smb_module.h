//--------------------------------------------------------------------------
// Copyright (C) 2016-2016 Cisco and/or its affiliates. All rights reserved.
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
//
// dce2_smb_module.h author Rashmi Pitre <rrp@cisco.com>

#ifndef DCE2_SMB_MODULE_H
#define DCE2_SMB_MODULE_H

#include "dce2_common.h"
#include "framework/module.h"
#include "dce2_list.h"

struct SnortConfig;

#define DCE2_VALID_SMB_VERSION_FLAG_V1 1
#define DCE2_VALID_SMB_VERSION_FLAG_V2 2

enum dce2SmbFileInspection
{
    DCE2_SMB_FILE_INSPECTION_OFF = 0,
    DCE2_SMB_FILE_INSPECTION_ON,
    DCE2_SMB_FILE_INSPECTION_ONLY
};

enum dce2SmbFingerprintPolicy
{
    DCE2_SMB_FINGERPRINT_POLICY_NONE = 0,
    DCE2_SMB_FINGERPRINT_POLICY_CLIENT,
    DCE2_SMB_FINGERPRINT_POLICY_SERVER,
    DCE2_SMB_FINGERPRINT_POLICY_BOTH,
};

struct dce2SmbShare
{
    char* unicode_str;
    unsigned int unicode_str_len;
    char* ascii_str;
    unsigned int ascii_str_len;
};

struct dce2SmbProtoConf
{
    dce2CommonProtoConf common;
    uint16_t co_reassemble_threshold;
    dce2SmbFingerprintPolicy smb_fingerprint_policy;
    uint8_t smb_max_chain;
    uint8_t smb_max_compound;
    uint16_t smb_valid_versions_mask;
    dce2SmbFileInspection smb_file_inspection;
    int16_t smb_file_depth;
    DCE2_List* smb_invalid_shares;
};

class Dce2SmbModule : public Module
{
public:
    Dce2SmbModule();
    ~Dce2SmbModule();

    bool set(const char*, Value&, SnortConfig*) override;

    unsigned get_gid() const override
    {
        return GID_DCE2;
    }

    const RuleMap* get_rules() const override;
    const PegInfo* get_pegs() const override;
    PegCount* get_counts() const override;
    ProfileStats* get_profile(unsigned, const char*&, const char*&) const override;
    void get_data(dce2SmbProtoConf&);

private:
    dce2SmbProtoConf config;
};

void print_dce2_smb_conf(dce2SmbProtoConf& config);

#endif

