#include <cstdlib>
#include <Rcpp.h>
using namespace Rcpp;

// This is a simple example of exporting a C++ function to R. You can
// source this function into an R session using the Rcpp::sourceCpp 
// function (or via the Source button on the editor toolbar). Learn
// more about Rcpp at:
//
//   http://www.rcpp.org/
//   http://adv-r.had.co.nz/Rcpp.html
//   http://gallery.rcpp.org/
//

// [[Rcpp::export(".parseYOYI")]]
List parseYOYI(CharacterVector x, char split) {
  IntegerVector click(x.size());
  IntegerVector price(x.size());
  
  // sparse matrix
  std::vector<int> mj, mp(x.size() + 1, 0), buf;
  
  SEXP px = wrap(x);
  for(::R_len_t i = 0;i < x.size();i++) {
    char* s1 = const_cast<char*>(CHAR(STRING_ELT(px, i)));
    char* s2(NULL);
    click[i] = strtol(s1, &s2, 10);
    s1 = strchr(s2, split) + 1;
    price[i] = strtol(s1, &s2, 10);
    s1 = s2 + 1;
    int &p(mp[i + 1]);
    buf.clear();
    while(*s1 != 0) {
      int feature = strtol(s1, &s1, 10);
      int value = strtol(s1 + 1, &s1, 10);
      if (value != 1) throw std::runtime_error("value is not 1");
      buf.push_back(feature);
      p++;
    }
    std::sort(buf.begin(), buf.end());
    mj.insert(mj.end(), buf.begin(), buf.end());
    p += mp[i];
  }
  return List::create(
    Named("click") = click, 
    Named("price") = price,
    Named("j") = wrap(mj),
    Named("p") = wrap(mp)
    );
}


// You can include R code blocks in C++ files processed with sourceCpp
// (useful for testing and development). The R code will be automatically 
// run after the compilation.
//

/*** R
library(methods)
library(Matrix)
parseYOYI <- function(x, split = "\t") {
  stopifnot(is.character(x))
  stopifnot(is.character(split))
  stopifnot(length(split) == 1)
  
  .r <- .parseYOYI(x, split)
  list(
    click = .r$click,
    price = .r$price,
    X = new("ngRMatrix", p = .r$p, j = .r$j, Dim = c(length(x), max(.r$j) + 1L))
  )
}

x <- c("0\t1\t1:1\t2:1\t3:1", "0\t25\t2:1\t4:1")
parseYOYI(x)
*/
