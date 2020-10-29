// SPDX-License-Identifier: Apache-2.0

// Implementations of string callbacks
#include <limits>
#include <iostream>

// Define function pointer we will use later
std::istream* (*file_callback)(std::string, std::iostream&);

// Define a factory class for building any buffer of text
class GB_callback {
  char *callback_string;
  char *include_filename;
  public:
     void load_string(char *fname, char *input){
        callback_string = input; 
        include_filename =  fname;
     }

     std::istream* callback( std::string filename, std::iostream& tmp_stream) {
        if ( filename == std::string(include_filename) ) {
           tmp_stream << callback_string; 
           return &tmp_stream;
        }
        else {
           return 0;
        } 
     }
};


//Semi-ring callbacks

std::istream* semiring_plus_times_callback( std::string filename, std::iostream& tmp_stream);

std::istream* semiring_min_plus_callback( std::string filename, std::iostream& tmp_stream);

std::istream* semiring_max_plus_callback( std::string filename, std::iostream& tmp_stream);


//Monoid callbacks

std::istream* file_callback_plus(std::string filename, std::iostream& tmp_stream);

std::istream* file_callback_max(std::string filename, std::iostream& tmp_stream);

std::istream* file_callback_min(std::string filename, std::iostream& tmp_stream);


std::istream* semiring_plus_times_callback( std::string filename, 
                                             std::iostream& tmp_stream)
{
  if (filename == "mySemiRing.h") {
    tmp_stream << "#define MONOID_IDENTITY (T_Z)0\n"
                  "#define MULADD( c, a, b ) (c) += (T_Z)( (a) * (b) )\n" 
                  "#define MUL( a, b) (a) * (b)\n"
                  "#define ADD( a, b) (a) + (b)\n";
    return &tmp_stream;
  }
  else {
    // Find this file through other mechanisms
    return 0;
  }

}

std::istream* semiring_min_plus_callback( std::string filename, 
                                           std::iostream& tmp_stream)
{ // Define the identity and operations for the (MIN,PLUS) semi-ring. mul->+, add -> min
  if (filename == "mySemiRing.h") {
    tmp_stream << "#define MONOID_IDENTITY std::numeric_limits<T_Z>::max()\n"
                  "#define MUL( a, b) (a) + (b)\n"
                  "#define ADD( a, b) (a) < (b) ? (a) : (b)\n";
    return &tmp_stream;
  }
  else {
    // Find this file through other mechanisms
    return 0;
  }

}

std::istream* semiring_max_plus_callback( std::string filename, 
                                           std::iostream& tmp_stream)
{ // Define the identity and operations for the (MAX,PLUS) semi-ring. mul->+, add -> max
  if (filename == "mySemiRing.h") {
    tmp_stream << "#define MONOID_IDENTITY std::numeric_limits<T_Z>::min()\n"
                  "#define MUL( a, b) (a) + (b)\n"
                  "#define ADD( a, b) (a) > (b) ? (a) : (b)\n";
    return &tmp_stream;
  }
  else {
    // Find this file through other mechanisms
    return 0;
  }

}

std::istream* file_callback_plus(std::string filename, std::iostream& tmp_stream) {
  // User returns NULL or pointer to stream containing file source
  // Note: tmp_stream is provided for convenience
  if (filename == "myOp.h") {
    tmp_stream << "#pragma once\n"
                  "#define MONOID_IDENTITY (T)0\n"
                  "#define OP( a, b) (a) + (b)\n";
    return &tmp_stream;
  }
  else {
    // Find this file through other mechanisms
    return 0;
  }
}

std::istream* file_callback_max(std::string filename, std::iostream& tmp_stream) {
  // User returns NULL or pointer to stream containing file source
  // Note: tmp_stream is provided for convenience
  if (filename == "myOp.h") {
    tmp_stream << "#pragma once\n"
                  "#include <limits>\n"
                  "#define MONOID_IDENTITY std::numeric_limits<T>::min()\n"
                  "#define OP( a, b) (a) > (b) ? (a) : (b)\n";

    return &tmp_stream;
  }
  else {
    // Find this file through other mechanisms
    return 0;
  }
}

std::istream* file_callback_min(std::string filename, std::iostream& tmp_stream) {
  // User returns NULL or pointer to stream containing file source
  // Note: tmp_stream is provided for convenience
  if (filename == "myOp.h") {
    tmp_stream << "#pragma once\n"
                  "#include <limits>\n"
                  "#define MONOID_IDENTITY std::numeric_limits<T>::max()\n"
                  "#define OP( a, b) (a) < (b) ? (a) : (b)\n";

    return &tmp_stream;
  }
  else {
    // Find this file through other mechanisms
    return 0;
  }
}


