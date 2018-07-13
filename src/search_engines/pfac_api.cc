#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "framework/mpse.h"

#include "pfac/pfac.h"

//-------------------------------------------------------------------------
// "pfac"
//-------------------------------------------------------------------------

class Pfac : public Mpse
{
private:
    PFAC_STRUCT * obj;

public:
    Pfac(SnortConfig*, const MpseAgent* agent)
        : Mpse("pfac", false)
    { obj = pfacNew(agent); }

    ~Pfac()
    { pfacFree(obj); }

    int add_pattern(
        SnortConfig*, const uint8_t* P, unsigned m,
        const PatternDescriptor& desc, void* user) override
    {
        return pfacAddPattern(obj, P, m, desc.no_case, desc.negated, user);
    }

    int prep_patterns(SnortConfig* sc) override
    { return pfacCompile(sc, obj); }

    int _search(
        const uint8_t* T, int n, MpseMatch match,
        void* context, int* current_state) override
    {
        return pfacSearch(obj, T, n, match, context, current_state);
    }

    int print_info() override
    { return pfacPrintDetailInfo(obj); }

    int get_pattern_count() override
    { return pfacPatternCount(obj); }
};

//-------------------------------------------------------------------------
// api
//-------------------------------------------------------------------------

static Mpse* pfac_ctor(
    SnortConfig* sc, class Module*, bool use_gc, const MpseAgent* agent)
{
    return new Pfac(sc, agent);
}

static void pfac_dtor(Mpse* p)
{
    delete p;
}

static void pfac_init()
{
    pfac_init_xlatcase();
}

static void pfac_print()
{
    pfacPrintSummaryInfo();
}

static const MpseApi pfac_api =
{
    {
        PT_SEARCH_ENGINE,
        sizeof(MpseApi),
        SEAPI_VERSION,
        0,
        API_RESERVED,
        API_OPTIONS,
        "pfac",
        "Parallel Failureless Aho-Corasick on GPU",
        nullptr,
        nullptr
    },
    false,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    pfac_ctor,
    pfac_dtor,
    pfac_init,
    pfac_print,
};

#ifdef BUILDING_SO
SO_PUBLIC const BaseApi* snort_plugins[] =
#else
const BaseApi* se_pfac[] =
#endif
{
    &pfac_api.base,
    nullptr
};

