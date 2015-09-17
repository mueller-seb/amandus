// This direcotry contains all the implementaion files

#include <hello.h>
#include <iostream>

namespace Salutation
{
void Hello::deutsch_hello()
{
  std::cout<<"Hallo! ich bin hier :)"<<std::endl;
}

void Hello::english_hello()
{
  std::cout<<"Hello! I am here :)"<<std::endl;
}

void Hello::italian_hello()
{
  std::cout<<"Bon giorno! Ciao! Io sono qui :)"<<std::endl;
}

}
