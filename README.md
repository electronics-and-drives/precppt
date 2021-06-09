# Serving PRECEPT Models

## Installation

Depends on:

- [LibTorch](https://pytorch.org/cppdocs/installing.html) 
- [yaml-cpp](https://github.com/jbeder/yaml-cpp)

## Build from Source

```sh
$ mkdir build
$ cd build
$ cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
$ cmake --build .
```

## TODO

- [ ] Overload constructor w/o `.yml` and use Setters.
    - [ ] Implement these Setters
- [ ] Implement data pre/post processing and transformations
- [ ] Change all memeber datatypes to `torch::Tensor` and convert in
  setter/getter to array.
