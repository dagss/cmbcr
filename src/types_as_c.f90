module types_as_c
  use iso_c_binding
  implicit none

  integer, parameter :: dp = c_double
  integer, parameter :: sp = c_float
  integer, parameter :: i1b = c_int8_t
  integer, parameter :: i4b = c_int32_t
  integer, parameter :: i8b = c_int64_t
  integer, parameter :: lgt = c_bool
  integer, parameter :: dpc = c_double_complex


end module
