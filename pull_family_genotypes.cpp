#include <sstream>
#include <string>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <array>
#include <time.h>
#include <stdio.h>
#include <bitset>
#include <functional>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/tokenizer.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string.hpp>

// Compile with:
// g++ pull_family_genotypes.cpp -I/usr/local/include -L/usr/local/lib -lboost_iostreams-mt -std=c++11 -O3 -o pull_family_genotypes

// On sherlock:
// module load gcc/4.8.1
// module load boost
// g++ pull_family_genotypes.cpp -I/usr/local/include -L/usr/local/lib -lboost_iostreams -std=c++11 -O3 -o pull_family_genotypes

// Run with:
// ./pull_family_genotypes ms1.22.vcf.gz v34.forCompoundHet.ped AU0918 AU0918.ms1.22.vcf
//

void pull_family_genotypes(std::string vcf_filename, std::string family_id, std::string ped_filename, std::string out_filename) {

    // Set up input vcf
    std::ifstream vcf_file(vcf_filename, std::ios_base::in | std::ios_base::binary);
    boost::iostreams::filtering_istream in_vcf;
    in_vcf.push(boost::iostreams::gzip_decompressor());
    in_vcf.push(vcf_file);

    // Pull individuals from ped
    std::string line;
    std::vector<std::string> pieces;
    std::ifstream ped_file(ped_filename, std::ios_base::in);
    std::vector<std::string> members;
    while(std::getline(ped_file, line)) {
        boost::split(pieces, line, boost::is_any_of("\t"));
        if(pieces[0] == family_id) {
            members.push_back(pieces[1]);
            members.push_back(pieces[2]);
            members.push_back(pieces[3]);
        }
    }
    ped_file.close();

    // Keep only unique members
    sort(members.begin(), members.end());
    members.erase(std::unique(members.begin(), members.end()), members.end());

    // Set up output file
    std::ofstream out_file(out_filename);
    boost::iostreams::filtering_ostream out_vcf;
    out_vcf.push(boost::iostreams::gzip_compressor());
    out_vcf.push(out_file);

    std::vector<int> member_indices(members.size());
    std::vector<std::string> genotypes(members.size());

    while(std::getline(in_vcf, line)) {
        if(boost::starts_with(line, "##")) {
            out_vcf << line << std::endl;
        }
        else if(boost::starts_with(line, "#")) {
            boost::split(pieces, line, boost::is_any_of("\t"));
            for(int i=9; i < pieces.size(); ++i) {
                auto it = std::find(members.begin(), members.end(), pieces[i]);
                if(it != members.end()) {
                    member_indices[std::distance(members.begin(), it)] = i;
                }
            }
            pieces.resize(9);
            out_vcf << boost::algorithm::join(pieces, "\t") << "\t";
            out_vcf << boost::algorithm::join(members, "\t") << std::endl;
        }
        else {
            boost::split(pieces, line, boost::is_any_of("\t"));
            std::transform(member_indices.begin(), member_indices.end(), 
                genotypes.begin(), [&](int index) {return pieces[index].substr(0, 3);});

            bool include = false;
            for(std::string genotype: genotypes) {
                if(genotype != "0/0" && genotype != "./.") {
                    include = true;
                    break;
                }
            }

            if(include) {
                pieces.resize(8);
                out_vcf << boost::algorithm::join(pieces, "\t") << "\t" << "GT\t";
                out_vcf << boost::algorithm::join(genotypes, "\t") << std::endl;
            }
        }
    }
    
    // Close files
    boost::iostreams::close(in_vcf);
    boost::iostreams::close(out_vcf);
}

int main(int argc, char *argv[]) {

    std::string vcf_filename = argv[1];
    std::string ped_filename = argv[2];
    std::string family_id = argv[3];
    std::string out_filename = argv[4];

    printf("Pulling family %s genotypes from %s using %s\n", family_id.c_str(), vcf_filename.c_str(), ped_filename.c_str());
    clock_t t = clock();
        
    pull_family_genotypes(vcf_filename, family_id, ped_filename, out_filename);

    t = clock() - t;
    printf("Finished in %f seconds\n", ((float)t)/CLOCKS_PER_SEC);

}






