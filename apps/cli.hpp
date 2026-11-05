#pragma once
// Tiny "--key value" command-line parser shared by the apps.

#include <cstdlib>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

class Args {
public:
    Args(int argc, char** argv) {
        for (int i = 1; i < argc; ++i) {
            std::string k = argv[i];
            if (k.rfind("--", 0) != 0) throw std::invalid_argument("unexpected argument '" + k + "'");
            k = k.substr(2);
            if (k == "help" || k == "h") {
                help_ = true;
                continue;
            }
            if (i + 1 >= argc) throw std::invalid_argument("missing value for --" + k);
            kv_[k] = argv[++i];
        }
    }
    bool help() const { return help_; }

    double get(const std::string& k, double def) { return has(k) ? std::stod(take(k)) : def; }
    int get(const std::string& k, int def) { return has(k) ? std::stoi(take(k)) : def; }
    std::string get(const std::string& k, const char* def) { return has(k) ? take(k) : std::string(def); }

    std::vector<int> get_ints(const std::string& k, std::vector<int> def) {
        if (!has(k)) return def;
        std::vector<int> v;
        std::stringstream ss(take(k));
        for (std::string item; std::getline(ss, item, ',');) v.push_back(std::stoi(item));
        return v;
    }
    std::vector<double> get_doubles(const std::string& k, std::vector<double> def) {
        if (!has(k)) return def;
        std::vector<double> v;
        std::stringstream ss(take(k));
        for (std::string item; std::getline(ss, item, ',');) v.push_back(std::stod(item));
        return v;
    }

    /// Call after reading every option: rejects typos.
    void finish() const {
        if (!kv_.empty()) throw std::invalid_argument("unknown option --" + kv_.begin()->first);
    }

private:
    bool has(const std::string& k) const { return kv_.count(k) != 0; }
    std::string take(const std::string& k) {
        std::string v = kv_.at(k);
        kv_.erase(k);
        return v;
    }
    std::map<std::string, std::string> kv_;
    bool help_ = false;
};
