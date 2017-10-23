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
// g++ split_by_family.cpp -I/usr/local/include -L/usr/local/lib -lboost_iostreams-mt -std=c++11 -O3 -o split_by_family

// On sherlock:
// module load gcc/4.8.1
// module load boost
// g++ split_by_family.cpp -I/usr/local/include -L/usr/local/lib -lboost_iostreams -std=c++11 -O3 -o split_by_family

// Run with:
// ./split_by_family ms1.22.vcf.gz v34.forCompoundHet.ped AU0918 AU0918.ms1.22.vcf
//


void split_by_family(std::string vcf_filename, std::string ped_filename, std::string out_directory) {

    // Set up input vcf
    std::ifstream vcf_file(vcf_filename, std::ios_base::in | std::ios_base::binary);
    boost::iostreams::filtering_istream in_vcf;
    in_vcf.push(boost::iostreams::gzip_decompressor());
    in_vcf.push(vcf_file);

    // Pull individuals from ped
    std::string line;
    std::vector<std::string> pieces;
    std::ifstream ped_file(ped_filename, std::ios_base::in);

    std::vector<std::string> family_ids;
    std::vector<std::vector<std::string>> family_members;

    // Read in ped file
    std::vector<std::string> ped_lines;
    while(std::getline(ped_file, line)) {
        ped_lines.push_back(line);
    }
    ped_file.close();

    // Sort lines so that families are ordered
    std::sort(ped_lines.begin(), ped_lines.end());

    for(std::string line: ped_lines) {
        boost::split(pieces, line, boost::is_any_of("\t"));
        std::string family_id = pieces[0];
        if(family_ids.size() == 0 || family_id != family_ids[family_ids.size()-1]) {
            // New family
            family_ids.push_back(family_id);
            std::vector<std::string> fm;
            fm.push_back(pieces[1]);
            fm.push_back(pieces[2]);
            fm.push_back(pieces[3]);
            family_members.push_back(fm);
        }
        else {
            std::vector<std::string>& fm = family_members[family_ids.size()-1];
            if(std::find(fm.begin(), fm.end(), pieces[1]) == fm.end()) {
                fm.push_back(pieces[1]);
            }
            if(std::find(fm.begin(), fm.end(), pieces[2]) == fm.end()) {
                fm.push_back(pieces[2]);
            }
            if(std::find(fm.begin(), fm.end(), pieces[3]) == fm.end()) {
                fm.push_back(pieces[3]);
            }
        }
    }

    std::cout << "Num families: " << family_ids.size() << std::endl;

    // Set up output files
    std::vector<std::ofstream> out_files(family_ids.size());
    std::vector<boost::iostreams::filtering_ostream> out_vcfs(family_ids.size());

    // Read vcf file
    std::vector<bool> include_family (family_ids.size(), false);
    std::vector<std::vector<int>> family_indices (family_ids.size());
    std::vector<std::vector<std::string>> family_genotypes (family_ids.size());

    std::stringstream vcf_header;
    int l = 0;
    while(std::getline(in_vcf, line)) {
        if(boost::starts_with(line, "##")) {
            // Add header to each file
            vcf_header << line << std::endl;
        }
        else if(boost::starts_with(line, "#")) {
            // Pull indices for each member
            std::unordered_map<std::string, int> member_to_index;
            boost::split(pieces, line, boost::is_any_of("\t"));
            for(int i=9; i < pieces.size(); ++i) {
                member_to_index[pieces[i]] = i;
            }
            pieces.resize(9);

            // Generate family indices
            int n = 0;
            for(int i=0; i < family_ids.size(); ++i) {
                std::vector<std::string> fm = family_members[i];
                std::sort(fm.begin(), fm.end());
                std::vector<std::string> fm_with_indices;
                for(std::string member: fm) {
                    if(member_to_index.count(member)) {
                        family_indices[i].push_back(member_to_index[member]);
                        family_genotypes[i].push_back("./.");
                        fm_with_indices.push_back(member);
                        include_family[i] = true;
                    }
                }

                // If at least one family member is in this vcf, set up output file
                if(include_family[i]) {
                    ++n;
                    out_files[i].open(out_directory + "/" + family_ids[i] + "." + vcf_filename);
                    out_vcfs[i].push(boost::iostreams::gzip_compressor());
                    out_vcfs[i].push(out_files[i]);
                    out_vcfs[i] << vcf_header.rdbuf();
                    out_vcfs[i] << boost::algorithm::join(pieces, "\t") << "\t";
                    out_vcfs[i] << boost::algorithm::join(fm_with_indices, "\t") << std::endl;
                }
            }
            std::cout << "Families indexed. " << std::endl;
            std::cout << "Remaining families: " << n << std::endl;
        }
        else {
            boost::split(pieces, line, boost::is_any_of("\t"));
            std::vector<std::string> header (pieces.begin(), pieces.begin()+8);
            std::string header_str = boost::algorithm::join(header, "\t") + "\tGT";

            for(int i = 0; i < family_ids.size(); ++i) {
                if(include_family[i]) {
                    std::stringstream new_line;
                    bool include_line = false;
                    for(int index: family_indices[i]) {
                        std::string genotype = pieces[index].substr(0, 3);
                        new_line << "\t" << genotype;
                        if(genotype != "./." && genotype != "0/0") {
                            include_line = true;
                        }
                    }
                    if(include_line) {
                        out_vcfs[i] << header_str << new_line.rdbuf() << std::endl;
                    }
                }
            }

            ++l;
            if(l % 1000000 == 0) {
                std::cout << "Lines complete: " << l << std::endl;
            }
        }
    }
    
    // Close files
    boost::iostreams::close(in_vcf);
    for(int i = 0; i < family_ids.size(); ++i) {
        if(include_family[i]) {
            boost::iostreams::close(out_vcfs[i]);
        }
    }
}

int main(int argc, char *argv[]) {

    std::string vcf_filename = argv[1];
    std::string ped_filename = argv[2];
    std::string out_directory = argv[3];

    printf("Splitting families from %s using %s\n", vcf_filename.c_str(), ped_filename.c_str());
    clock_t t = clock();
        
    split_by_family(vcf_filename, ped_filename, out_directory);

    t = clock() - t;
    printf("Finished in %f seconds\n", ((float)t)/CLOCKS_PER_SEC);

}






