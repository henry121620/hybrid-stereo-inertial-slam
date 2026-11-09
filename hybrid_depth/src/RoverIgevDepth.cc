#include "RoverIgevDepth.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace ORB_SLAM3
{
namespace RoverIgevDepth
{

struct DepthIndex
{
    bool initialized = false;
    bool enabled = false;
    std::string dir;
    long long timestamp_offset_ns = 0;
    std::map<long long, std::string> ts_to_file;
};

static DepthIndex& GetIndex()
{
    static DepthIndex index;
    return index;
}

static bool EnvFlagEnabled(const char* name)
{
    const char* v = std::getenv(name);
    if (!v) return false;

    std::string s(v);
    return !(s == "0" || s == "false" || s == "FALSE" ||
             s == "off" || s == "OFF" || s == "no" || s == "NO");
}

bool Enabled()
{
    return EnvFlagEnabled("ROVER_USE_IGEV_DEPTH");
}

static bool EndsWith(const std::string& s, const std::string& suffix)
{
    if (s.size() < suffix.size()) return false;
    return s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static bool ParseTimestampFromFilename(const std::string& name, long long& ts)
{
    if (!EndsWith(name, ".npy")) return false;

    std::string stem = name.substr(0, name.size() - 4);
    if (stem.empty()) return false;

    for (char c : stem)
    {
        if (c < '0' || c > '9') return false;
    }

    try
    {
        ts = std::stoll(stem);
        return true;
    }
    catch (...)
    {
        return false;
    }
}

static void InitIndex()
{
    DepthIndex& index = GetIndex();
    if (index.initialized) return;

    index.initialized = true;
    index.enabled = Enabled();

    if (!index.enabled)
    {
        std::cout << "[IGEV depth] disabled" << std::endl;
        return;
    }

    const char* envDir = std::getenv("ROVER_IGEV_DEPTH_DIR");
    if (!envDir)
    {
        std::cerr << "[IGEV depth] ERROR: ROVER_IGEV_DEPTH_DIR is not set" << std::endl;
        index.enabled = false;
        return;
    }

    index.dir = std::string(envDir);

    if (const char* envOffset = std::getenv("ROVER_IGEV_TIMESTAMP_OFFSET_NS"))
    {
        try
        {
            index.timestamp_offset_ns = std::stoll(envOffset);
        }
        catch (...)
        {
            std::cerr << "[IGEV depth] WARN: invalid ROVER_IGEV_TIMESTAMP_OFFSET_NS="
                      << envOffset << ", using 0" << std::endl;
            index.timestamp_offset_ns = 0;
        }
    }

    DIR* dp = opendir(index.dir.c_str());
    if (!dp)
    {
        std::cerr << "[IGEV depth] ERROR: cannot open dir: " << index.dir << std::endl;
        index.enabled = false;
        return;
    }

    struct dirent* entry;
    while ((entry = readdir(dp)) != nullptr)
    {
        std::string name(entry->d_name);
        long long ts = 0;
        if (!ParseTimestampFromFilename(name, ts)) continue;

        index.ts_to_file[ts] = index.dir + "/" + name;
    }

    closedir(dp);

    std::cout << "[IGEV depth] enabled" << std::endl;
    std::cout << "[IGEV depth] dir: " << index.dir << std::endl;
    std::cout << "[IGEV depth] timestamp offset ns: " << index.timestamp_offset_ns << std::endl;
    std::cout << "[IGEV depth] indexed " << index.ts_to_file.size() << " NPY files" << std::endl;

    if (!index.ts_to_file.empty())
    {
        std::cout << "[IGEV depth] first ts: " << index.ts_to_file.begin()->first << std::endl;
        std::cout << "[IGEV depth] last  ts: " << index.ts_to_file.rbegin()->first << std::endl;
    }
}

static bool ReadNpyFloat32(const std::string& path, cv::Mat& out)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open())
    {
        std::cerr << "[IGEV depth] ERROR: cannot open npy: " << path << std::endl;
        return false;
    }

    char magic[6];
    f.read(magic, 6);
    if (!f || std::memcmp(magic, "\x93NUMPY", 6) != 0)
    {
        std::cerr << "[IGEV depth] ERROR: invalid npy magic: " << path << std::endl;
        return false;
    }

    unsigned char major = 0, minor = 0;
    f.read(reinterpret_cast<char*>(&major), 1);
    f.read(reinterpret_cast<char*>(&minor), 1);

    uint32_t header_len = 0;
    if (major == 1)
    {
        uint16_t hl16 = 0;
        f.read(reinterpret_cast<char*>(&hl16), 2);
        header_len = hl16;
    }
    else if (major == 2 || major == 3)
    {
        uint32_t hl32 = 0;
        f.read(reinterpret_cast<char*>(&hl32), 4);
        header_len = hl32;
    }
    else
    {
        std::cerr << "[IGEV depth] ERROR: unsupported npy version: " << int(major) << "." << int(minor) << std::endl;
        return false;
    }

    std::string header(header_len, '\0');
    f.read(&header[0], header_len);

    const bool is_f4 = header.find("'descr': '<f4'") != std::string::npos ||
                       header.find("\"descr\": \"<f4\"") != std::string::npos ||
                       header.find("'descr': '|f4'") != std::string::npos;

    if (!is_f4)
    {
        std::cerr << "[IGEV depth] ERROR: only float32 npy is supported. header=" << header << std::endl;
        return false;
    }

    if (header.find("True") != std::string::npos)
    {
        std::cerr << "[IGEV depth] ERROR: Fortran-order npy is not supported. header=" << header << std::endl;
        return false;
    }

    size_t p0 = header.find('(');
    size_t p1 = header.find(')', p0);
    if (p0 == std::string::npos || p1 == std::string::npos)
    {
        std::cerr << "[IGEV depth] ERROR: cannot parse shape. header=" << header << std::endl;
        return false;
    }

    std::string shape = header.substr(p0 + 1, p1 - p0 - 1);
    std::replace(shape.begin(), shape.end(), ',', ' ');

    std::stringstream ss(shape);
    int rows = 0, cols = 0;
    ss >> rows >> cols;

    if (rows <= 0 || cols <= 0)
    {
        std::cerr << "[IGEV depth] ERROR: invalid shape. header=" << header << std::endl;
        return false;
    }

    cv::Mat depth(rows, cols, CV_32F);
    const size_t bytes = static_cast<size_t>(rows) * static_cast<size_t>(cols) * sizeof(float);

    f.read(reinterpret_cast<char*>(depth.data), bytes);
    if (!f)
    {
        std::cerr << "[IGEV depth] ERROR: failed to read data: " << path << std::endl;
        return false;
    }

    out = depth;
    return true;
}

static bool FindNearestFile(const double timestampSec, std::string& path, long long& matchedTs, long long& queryTs)
{
    DepthIndex& index = GetIndex();
    if (!index.initialized) InitIndex();
    if (!index.enabled || index.ts_to_file.empty()) return false;

    queryTs = static_cast<long long>(std::llround(timestampSec * 1e9)) +
              index.timestamp_offset_ns;

    auto it = index.ts_to_file.lower_bound(queryTs);

    long long bestDiff = 9223372036854775807LL;
    auto bestIt = index.ts_to_file.end();

    if (it != index.ts_to_file.end())
    {
        long long diff = std::llabs(it->first - queryTs);
        if (diff < bestDiff)
        {
            bestDiff = diff;
            bestIt = it;
        }
    }

    if (it != index.ts_to_file.begin())
    {
        auto it2 = it;
        --it2;
        long long diff = std::llabs(it2->first - queryTs);
        if (diff < bestDiff)
        {
            bestDiff = diff;
            bestIt = it2;
        }
    }

    // EuRoC camera interval is tens of ms. This threshold only rejects clearly wrong matching.
    const long long maxAllowedNs = 5000000LL; // 5 ms
    if (bestIt == index.ts_to_file.end() || bestDiff > maxAllowedNs)
    {
        static int warn_count = 0;
        if (warn_count < 10)
        {
            std::cerr << "[IGEV depth] WARN: no close depth for timestamp "
                      << queryTs << ", nearest diff ns=" << bestDiff << std::endl;
            warn_count++;
        }
        return false;
    }

    matchedTs = bestIt->first;
    path = bestIt->second;
    return true;
}

bool LoadDepthForTimestamp(const double timestampSec, cv::Mat& depth)
{
    if (!Enabled()) return false;

    std::string path;
    long long matchedTs = 0;
    long long queryTs = 0;

    if (!FindNearestFile(timestampSec, path, matchedTs, queryTs)) return false;

    bool ok = ReadNpyFloat32(path, depth);
    if (!ok) return false;

    static int print_count = 0;
    if (print_count < 5)
    {
        std::cout << "[IGEV depth] loaded: query_ts=" << queryTs
                  << " matched_ts=" << matchedTs
                  << " shape=" << depth.cols << "x" << depth.rows
                  << " file=" << path << std::endl;
        print_count++;
    }

    return true;
}

}
}
