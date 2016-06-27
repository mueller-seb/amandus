// Here you put your files containing main method

#include <hello.h>
#include <iostream>
#include <parameters.h>

using namespace Salutation;
int
main()
{
  dealii::ParameterHandler param;
  Parameters::declare_parameters(param);
  param.read_input("hello_main.prm", true);
  Hello hi;
  switch (param.get_integer("language"))
  {
    case 0:
      hi.deutsch_hello();
      break;
    case 1:
      hi.english_hello();
      break;
    case 2:
      hi.italian_hello();
      break;
  }
  return 0;
}
