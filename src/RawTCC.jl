module Rawx

using Blio
using HDF5
using ERFA
using Geodesy
using EarthOrientation
using RadioInterferometry
# For package version info (for UVH5 "history" header)
import Pkg
import UUIDs: UUID
import YAML

# @assert xgpuinfo.npol == 2 "xGPU must be compiled with NPOL=2"

"""
Julian date corresponding to MJD 0
"""
const MJD0 = ERFA.DJM0 # 2400000.5

"""
Julia date corresponding to the J2000 epoch (2000-01-01T12:00:00)
"""
const J2000 = ERFA.DJ00 # 2451545.0

"""
POLMAP is a Dict{Any, Any} that maps polarization to UVFITS integer and UVFITS
integer to polarization.  Polarization keys may be given as a single symbol
(e.g. `:i` or `:xx`), a single string (e.g. `"i"` or `"xx"`), a
Tuple{Symbol,Symbol}, a Tuple{String,String}, or and Int.  Uppercase and
lowercase are supported, but not mixed-case.  Lowercase is preferred.
Polarization Symbol values (Stokes parameters only) are lower case.
Polarization Tuple values (cross polarization only) are always lower case
Symbols.
"""
const POLMAP = Dict(
  # Symbol keys
  :i  =>  1, :q  =>  2, :u  =>  3,  :v =>  4,
  :I  =>  1, :Q  =>  2, :U  =>  3,  :V =>  4,
  :rr => -1, :ll => -2, :rl => -3, :lr => -4,
  :RR => -1, :LL => -2, :RL => -3, :LR => -4,
  :xx => -5, :yy => -6, :xy => -7, :yx => -8,
  :XX => -5, :YY => -6, :XY => -7, :YX => -8,
  # String keys
  "i"  =>  1, "q"  =>  2, "u"  =>  3,  "v" =>  4,
  "I"  =>  1, "Q"  =>  2, "U"  =>  3,  "V" =>  4,
  "rr" => -1, "ll" => -2, "rl" => -3, "lr" => -4,
  "RR" => -1, "LL" => -2, "RL" => -3, "LR" => -4,
  "xx" => -5, "yy" => -6, "xy" => -7, "yx" => -8,
  "XX" => -5, "YY" => -6, "XY" => -7, "YX" => -8,
  # Tuple{Symbol,Symbol} keys
  (:r,:r) => -1, (:l,:l) => -2, (:r,:l) => -3, (:l,:r) => -4,
  (:R,:R) => -1, (:L,:L) => -2, (:R,:L) => -3, (:L,:R) => -4,
  (:x,:x) => -5, (:y,:y) => -6, (:x,:y) => -7, (:y,:x) => -8,
  (:X,:X) => -5, (:Y,:Y) => -6, (:X,:Y) => -7, (:Y,:X) => -8,
  # Tuple{String,String} keys
  ("r","r") => -1, ("l","l") => -2, ("r","l") => -3, ("l","r") => -4,
  ("R","R") => -1, ("L","L") => -2, ("R","L") => -3, ("L","R") => -4,
  ("x","x") => -5, ("y","y") => -6, ("x","y") => -7, ("y","x") => -8,
  ("X","X") => -5, ("Y","Y") => -6, ("X","Y") => -7, ("Y","X") => -8,
  # Int keys (reverse map)
  1 => :i, 2 => :q, 3 => :u, 4 => :v,
  -1 => (:r,:r), -2 => (:l,:l), -3 => (:r,:l), -4 => (:l,:r),
  -5 => (:x,:x), -6 => (:y,:y), -7 => (:x,:y), -8 => (:y,:x)
)

"""
`UVH5BoolDatatype` is an `HDF5.Datatype` that should be used when creating the
`Data/flags` dataset of a UVH5 file.

Per the UVH5 specification, the boolean flags stored in the `/Data/flags`
dataset of a UVH5 file are not the HDF5-provided H5T_NATIVE_HBOOL type.
Instead, they are defined to conform to the h5py implementation of the numpy
boolean type.  More specifically, the `Data/flags` dataset uses an HDF5 enum
datatype to encode the numpy boolean type.
"""
const UVH5BoolDatatype = HDF5.Datatype(0)

function __init__()
  UVH5BoolDatatype.id = HDF5.h5t_create(HDF5.H5T_ENUM, sizeof(Bool))
  HDF5.h5t_enum_insert(UVH5BoolDatatype, "FALSE", Ref(false))
  HDF5.h5t_enum_insert(UVH5BoolDatatype, "TRUE", Ref(true))
end

"""
    struple(s::AbstractString, n::Int=sizeof(s), pad::UInt8=0x0)::NTuple{n,UInt8}
Return a tuple containing the UInt8 code points of `s` padded to length `n`
with `pad`.
"""
function struple(s::AbstractString, n::Int=sizeof(s), pad::UInt8=0x0)::NTuple{n,UInt8}
    Tuple([UInt8.(codeunits(s)); fill(pad, n-sizeof(s))])
end

"""
    uvh5_string_datatype(size;
                         strpad=HDF5.H5T_STR_NULLTERM,
                         cset=HDF5.H5T_CSET_UTF8)
Create an HDF5 fixed length string datatype. Padding specified by `strpad`,
character set specified by `cset`.
"""
function uvh5_string_datatype(size;
                              strpad=HDF5.H5T_STR_NULLTERM,
                              cset=HDF5.H5T_CSET_UTF8)
    dt = HDF5.create_datatype(HDF5.H5T_STRING, size)
    HDF5.h5t_set_strpad(dt, strpad)
    HDF5.h5t_set_cset(dt, cset)
    dt
end

"""
Create a fixed length string scalar HDF5 dataset `name` in `parent` using
padding and character set specified by `strpad` and `cset`. The string in
`string` is stored in this dataset.
"""
function uvh5_string_dataset(parent, name, string::AbstractString;
                             strpad=HDF5.H5T_STR_NULLTERM,
                             cset=HDF5.H5T_CSET_UTF8,
                             pv...)
    strsize = sizeof(string)
    dt = uvh5_string_datatype(strsize, strpad=strpad, cset=cset)
    ds = create_dataset(parent, name, dt, (); pv...)
    write_dataset(ds, dt, string)
    ds
end

"""
Create a fixed length string array HDF5 dataset `name` in `parent` using
padding and character set specified by `strpad` and `cset`. The strings in
`strings` are stored in this dataset.
"""
function uvh5_string_dataset(parent, name,
                             strings::AbstractArray{<:AbstractString};
                             strpad=HDF5.H5T_STR_NULLTERM,
                             cset=HDF5.H5T_CSET_UTF8,
                             pv...)
    maxsize = maximum(sizeof, strings)
    pad = strpad == HDF5.H5T_STR_SPACEPAD ? 0x20 : 0x00
    dt = uvh5_string_datatype(maxsize, strpad=strpad, cset=cset)
    ds = create_dataset(parent, name, dt, size(strings); pv...)
    strings_tuple = struple.(strings, maxsize, pad)
    write_dataset(ds, dt, strings_tuple)
    ds
end

"""
Type for optional parameters: `Union{T, Nothing}`
"""
Optional{T} = Union{T, Nothing}

"""
    frame_transform(coords, translate)
    frame_transform(coords, rotate)
    frame_transform(coords, translate, rotate)
    frame_transform(coords, rotate, translate)

Convert coordinates `coords` from one frame to another through a series of
operations. `coords` is an Array with one "point" per column. The two types
of supported operations are: `translate` and `rotate`. `translate` is a
Vector that will be added to each column (i.e. "point"). `rotate` is a
rotation Matrix with as many columns as there are rows in `coords`.

These operations support common radio astronomy coordinate frame
transformations such as:

    | From -> To   | Translate |  Rotate  | Translate |
    |:------------:|:---------:|:--------:|:---------:|
    | ecef -> enu  |  -refpos  |  xyz2enu |           |
    | ecef -> xyz  |  -refpos  |          |           |
    | enu  -> ecef |           |  enu2xyz |  +refpos  |
    | enu  -> xyz  |           |  enu2xyz |           |
    | xyz  -> ecef |  +refpos  |          |           |
    | xyz  -> enu  |           |  xyz2enu |           |
"""
function frame_transform(coords::AbstractVecOrMat{<:Real},
                         translate::AbstractVector{<:Real})
    coords .+ translate
end

function frame_transform(coords::AbstractVecOrMat{<:Real},
                         rotate::AbstractArray{<:Real,2})
    rotate * coords
end

function frame_transform(coords::AbstractVecOrMat{<:Real},
                         rotate::AbstractArray{<:Real,2},
                         translate::AbstractVector{<:Real})
    rotate * coords .+ translate
end

function frame_transform(coords::AbstractVecOrMat{<:Real},
                         translate::AbstractVector{<:Real},
                         rotate::AbstractArray{<:Real,2})
    rotate * (coords .+ translate)
end

"""
    getantpos(antennas::AbstractArray{<:AbstractDict{Symbol,Any},1},
              latitude::Union{Real,AbstractString},
              longitude::Union{Real,AbstractString},
              altitude::Real,
              to_frame::Optional{Symbol},
              from_frame::Optional{Symbol}=nothing
             )::Array{Float64,2}

Gets the antenna positions from `antennas` and converts to the requested
reference frame given by `to_frame`, which must be `:enu`, `:xyz`, `:ecef`, or
`nothing`.

Each element of the `antennas` Array is a Dicts that has a `:position` key
corresponding to a 3 element Array of floating point numbers.  The reference
frame of the antenna positions in `antennas` can be specified by `from_frame`,
which must be the Symbol `:enu` for the East-North-Up reference frame,
`:ecef` for the Earth-Centered-Earth-Fixed ITRF reference frame, `:xyz` for
the topocentric ITRF-aligned reference frame, or `nothing` in which case
`from_frame` is auto-detected by the geometric length of the first antenna
position vector.  If the position of the first antenna is less than 6e6 meters
from the origin then `from_frame` is assumed to be `:enu`, otherwise `:ecef`.

Conversion between ENU and ECEF frames requires `latitude`, `longitude`, and
`altitude` of the topocentric origin.  Passing `to_frame=nothing` will return
the values from `antennas` without any conversion.
"""
function getantpos(antennas::AbstractArray{<:AbstractDict{Symbol,Any},1},
                   latitude::Union{Real,AbstractString},
                   longitude::Union{Real,AbstractString},
                   altitude::Real,
                   to_frame::Optional{Symbol},
                   from_frame::Optional{Symbol}=nothing
                  )::Array{Float64,2}

  if !(to_frame in [:enu, :xyz, :ecef, nothing])
    error("invalid to_frame $frame)")
  end

  antpos = getindex.(antennas, :position)
  dist = hypot(antpos[1]...)

  if isnothing(from_frame)
    from_frame = dist < 6e6 ? :enu : :ecef
    @warn("from_frame not specified, assuming $from_frame")
  elseif from_frame == :enu
    dist < 6e6 || @warn("antenna position suggests ECEF, not ENU")
  elseif from_frame == :xyz
    dist < 6e6 || @warn("antenna position suggests ECEF, not XYZ")
  elseif from_frame == :ecef
    dist >= 6e6 || @warn("antenna position suggests ENU or XYZ, not ECEF")
  else
    error("invalid from_frame $from_frame)")
  end

  # Six supported frame conversions.
  #
  # From -> To     Translate -> Rotate -> Translate
  # ecef -> enu     -refpos     xyz2enu       0
  # ecef -> xyz     -refpos        1          0
  # enu  -> ecef       0        enu2xyz    +refpos
  # enu  -> xyz        0        enu2xyz       0
  # xyz  -> ecef       0           1       +refpos
  # xyz  -> enu        0        xyz2enu       0
  if (from_frame, to_frame) == (:ecef, :enu)
    translate = -ecef(lla(dms2deg(latitude), dms2deg(longitude), altitude), wgs84)
    rotate = xyz2enu(dms2rad(latitude), dms2rad(longitude))
    antpos = frame_transform.(antpos, Ref(translate), Ref(rotate))
  elseif (from_frame, to_frame) == (:ecef, :xyz)
    translate = -ECEF(LLA(dms2deg(latitude), dms2deg(longitude), altitude), wgs84)
    antpos = frame_transform.(antpos, Ref(translate))
  elseif (from_frame, to_frame) == (:enu, :ecef)
    rotate = enu2xyz(dms2rad.(latitude), dms2rad(longitude))
    translate = ecef(lla(dms2deg(latitude), dms2deg(longitude), altitude), wgs84)
    antpos = frame_transform.(antpos, Ref(rotate), Ref(translate))
  elseif (from_frame, to_frame) == (:enu, :xyz)
    rotate = enu2xyz(dms2rad(latitude), dms2rad(longitude))
    antpos = frame_transform.(antpos, Ref(rotate))
  elseif (from_frame, to_frame) == (:xyz, :ecef)
    translate = ecef(lla(dms2deg(latitude), dms2deg(longitude), altitude), wgs84)
    antpos = frame_transform.(antpos, Ref(translate))
  elseif (from_frame, to_frame) == (:xyz, :enu)
    rotate = xyz2enu(dms2rad(latitude), dms2rad(longitude))
    antpos = frame_transform.(antpos, Ref(rotate))
  end

  return float(reduce(hcat,antpos))
end

# Version for `from_frame` being a string
function getantpos(antennas::AbstractArray{<:AbstractDict{Symbol,Any},1},
                   latitude::Union{Real,AbstractString},
                   longitude::Union{Real,AbstractString},
                   altitude::Real,
                   to_frame::Optional{Symbol},
                   from_frame::AbstractString
                  )::Array{Float64,2}
  from_frame_sym = Symbol(lowercase(from_frame))
  getantpos(antennas, latitude, longitude, altitude, to_frame, from_frame_sym)
end

"""
    getantpos(telinfo::Dict{Symbol,Any}, to_frame::Optional{Symbol})::Array{Float64,2}

Gets the antenna positions from `telinfo[:antennas]` and converts to the
requested reference frame given by `to_frame`, which must be `:enu`,`:ecef`, or
`nothing`.  Additional fields from `telinfo` are used:

- `:latitude`
- `:longitude`
- `:altitude`
- `:antenna_position_frame` (passed on as `from_frame`, if present)
"""
function getantpos(telinfo::Dict{Symbol,Any},
                   to_frame::Optional{Symbol})::Array{Float64,2}
  antennas = telinfo[:antennas]
  latitude = get(telinfo, :latitude, 0)
  longitude = get(telinfo, :longitude, 0)
  altitude = get(telinfo, :altitude, 0)
  from_frame = get(telinfo, :antenna_position_frame, nothing)

  getantpos(antennas, latitude, longitude, altitude, to_frame, from_frame)
end

"""
    getpkgverinfo()::Dict{String,VersionNumber}

Get package and dependent package names and versions as a
`Dict{String,VersionNumber}`.
"""
function getpkgverinfo()::Dict{String,VersionNumber}
	toml = Pkg.TOML.parsefile(joinpath(pkgdir(@__MODULE__), "Project.toml"))
  pkgverinfo = Dict(toml["name"] => VersionNumber(toml["version"]))
  pkgdeps = Pkg.dependencies()
	foreach(toml["deps"]) do (name, uuid)
    version = pkgdeps[UUID(uuid)].version
    # Skip packages that have no version info (i.e. stdlibs)
    isnothing(version) && return # return from "do" function
    pkgverinfo[name] = version
	end
  pkgverinfo
end

"""
    uvh5create(name::AbstractString; swmr=false, metadata...)::HDF5.File

Create a UVH5 file and initialize/populate it based on values in `metadata`.
TODO: Provide docstrings for all kwarg parameters.
"""
function uvh5create(name::AbstractString;
                    swmr=false,
                    antennas::AbstractArray{<:AbstractDict{Symbol,Any},1},
                    input_map::AbstractArray{<:AbstractArray{String,1},1},
                    phase_type::AbstractString="phased",
                    latitude::Union{Real,AbstractString},
                    longitude::Union{Real,AbstractString},
                    altitude::Real,
                    telescope_name::AbstractString,
                    instrument::AbstractString,
                    object_name::AbstractString,
                    freq_array::AbstractVector{<:Real},
                    channel_width::AbstractVector{<:Real},
                    polarization_array::AbstractArray{Int,1},
                    # Optional parameters
                    # flex_spw_id_array defaults to single spectral window
                    flex_spw_id_array::Optional{AbstractVector{<:Integer}}=nothing,
                    antenna_diameter::Real=0.0,
                    dut1::Optional{Real}=nothing,
                    earth_omega::Optional{Real}=nothing,
                    gst0::Optional{Real}=nothing,
                    rdate::Optional{AbstractString}=nothing,
                    timesys::Optional{AbstractString}=nothing,
                    x_orientation::Optional{AbstractString}=nothing,
                    uvplane_reference_time::Optional{Int}=nothing,
                    phase_center_ra::Optional{Real}=nothing,
                    phase_center_dec::Optional{Real}=nothing,
                    phase_center_epoch::Optional{Real}=nothing,
                    phase_center_frame::Optional{AbstractString}=nothing,
                    antenna_position_frame::Optional{AbstractString}=nothing,
                    extra_kwargs...
                   )::HDF5.File

  # Check for valid inputs
  if phase_type != "phased" && phase_type != "drift"
    error("unrecognized phase_type: $(phase_type)")
  end

  # Determine various dimensions and reused parameters
  nants_telescope = length(antennas)
  # nants_data is number of unique antennas in input_map
  nants_data = length(unique(getindex.(input_map, 1)))
  nbls = ((nants_data + 1) * nants_data) ÷ 2
  nfreqs = length(freq_array)
  # If flex_spw_id_array is given, use it to create spw_array
  # otherwise, just set spw_array to [1].
  spw_array = unique(something(flex_spw_id_array, [1]))
  nspw = length(spw_array)
  npols = length(polarization_array)

  # Create HDF5 file (destroys any existing contents)
  uvh5 = h5open(name, "w"; swmr=swmr)
  # Create "/Header" and "/Data" groups
  hdr = create_group(uvh5, "/Header")
  data = create_group(uvh5, "/Data")

  # Create header required parameters (i.e. datasets)
  # Currently done in order presented in UVH5 memo
  hdr["latitude", layout=HDF5.H5D_COMPACT] = dms2deg(latitude)
  hdr["longitude", layout=HDF5.H5D_COMPACT] = dms2deg(longitude)
  hdr["altitude", layout=HDF5.H5D_COMPACT] = Float64(altitude)
  uvh5_string_dataset(hdr, "telescope_name", string(telescope_name),
                      layout=HDF5.H5D_COMPACT)
  uvh5_string_dataset(hdr, "instrument", string(instrument),
                      layout=HDF5.H5D_COMPACT)
  uvh5_string_dataset(hdr, "object_name", string(object_name),
                      layout=HDF5.H5D_COMPACT)
  uvh5_string_dataset(hdr, "history", "# Created by Rawx using:\n" *
                      YAML.yaml(getpkgverinfo()), layout=HDF5.H5D_COMPACT)
  uvh5_string_dataset(hdr, "phase_type", string(phase_type),
                      layout=HDF5.H5D_COMPACT)
  hdr["Nants_data", layout=HDF5.H5D_COMPACT] = nants_data
  hdr["Nants_telescope", layout=HDF5.H5D_COMPACT] = nants_telescope
  # Create ant_{1,2}_arrays as empty, but open ended in the time dimension.
  # Chunk size is size per dump.
  create_dataset(hdr, "ant_1_array", Int, ((0,),(-1,)),
                 chunk=(nbls,), compress=3)
  create_dataset(hdr, "ant_2_array", Int, ((0,),(-1,)),
                 chunk=(nbls,), compress=3)
  uvh5_string_dataset(hdr, "antenna_names", string.(getindex.(antennas, :name)),
                      layout=HDF5.H5D_COMPACT)
  hdr["antenna_numbers", layout=HDF5.H5D_COMPACT] = Int.(getindex.(antennas, :number))
  hdr["Nbls", layout=HDF5.H5D_COMPACT] =  nbls
  hdr["Nblts", layout=HDF5.H5D_COMPACT] = 0 # Initial value
  hdr["Nfreqs", layout=HDF5.H5D_COMPACT] = nfreqs
  hdr["Npols", layout=HDF5.H5D_COMPACT] = npols
  hdr["Ntimes", layout=HDF5.H5D_COMPACT] =  0 # Initial value
  hdr["Nspws", layout=HDF5.H5D_COMPACT] = nspw

  create_dataset(hdr, "uvw_array", Float64, ((3,0),(3,-1)),
                 chunk=(3,nbls), compress=3)

  create_dataset(hdr, "time_array", Float64, ((0,),(-1,)),
                 chunk=(nbls,), compress=3)

  create_dataset(hdr, "integration_time", Float64, ((0,),(-1,)),
                 chunk=(nbls,), compress=3)

  hdr["freq_array"] = float(freq_array)
  hdr["channel_width", layout=HDF5.H5D_COMPACT] = convert(Vector{Float64}, channel_width)
  hdr["spw_array", layout=HDF5.H5D_COMPACT] = spw_array
  if flex_spw_id_array !== nothing
    hdr["flex_spw_id_array", layout=HDF5.H5D_COMPACT] = flex_spw_id_array
  end
  # The `flex_spw` dataset needs to be the funky UVH5BoolDatatype type
  flex_spw_ds = create_dataset(hdr, "flex_spw", UVH5BoolDatatype, layout=HDF5.H5D_COMPACT, ())
  write_dataset(flex_spw_ds, UVH5BoolDatatype, flex_spw_id_array !== nothing)
  hdr["version", layout=HDF5.H5D_COMPACT] = "1.0"
  hdr["polarization_array", layout=HDF5.H5D_COMPACT] = polarization_array
  hdr["antenna_positions"] = getantpos(antennas, dms2deg(latitude),
                                       dms2deg(longitude), altitude,
                                       :xyz, antenna_position_frame)

  # Optional parameters follow.  Two optional parameters, `antenna_diameters`
  # and `lst_array`, are non-scalars, so they are handled specially.  The rest
  # are handled in a loop and stored as the HDF5 equivalent of their Julia
  # types.

  # The `antenna_diameters` dataset gets set iff a default value is given in
  # `antenna_diameter` or all `antennas` entries have a `:diameter`
  # field.
  if antenna_diameter > 0 || all(haskey.(kntennas, :diameter))
    hdr["antenna_diameters"] = map(antennas) do ant
      Float64(get(ant, :diameter, antenna_diameter))
    end
  end

  # The `lst_array` dataset gets created like the `time_array` dataset
  create_dataset(hdr, "lst_array", Float64, ((0,),(-1,)),
                 chunk=(nbls,), compress=3)

  # TODO Figure out how to get eval(:kwarg) functionality (i.e. something more
  # clever than ("kwarg", kwarg).

  # If phase_type is "phased", the phase_center parameters are not optional, so
  # we ensure they exist here (but defer their actual creation to the loop that
  # creates other optional parameters as well).
  if phase_type == "phased"
    for (name, val) in [("phase_center_ra", phase_center_ra),
                        ("phase_center_dec", phase_center_dec),
                        ("phase_center_epoch", phase_center_epoch),
                        ("phase_center_frame", phase_center_frame)]
      if isnothing(val)
        error("$name is required when phase_type == \"phased\"")
      end
    end
  end

  for (conv, name, val) in [
                  (Float64, "dut1",                   dut1),
                  (Float64, "earth_omega",            earth_omega),
                  (Float64, "gst0",                   gst0),
                  (string,  "rdate",                  rdate),
                  (string,  "timesys",                timesys),
                  (string,  "x_orientation",          x_orientation),
                  (Int,     "uvplane_reference_time", uvplane_reference_time),
                  (Float64, "phase_center_ra",        phase_center_ra),
                  (Float64, "phase_center_dec",       phase_center_dec),
                  (Float64, "phase_center_epoch",     phase_center_epoch),
                  (string,  "phase_center_frame",     phase_center_frame)
                 ]
    if !isnothing(val)
      if conv == string
        uvh5_string_dataset(hdr, name, string(val), layout=HDF5.H5D_COMPACT)
      else
        hdr[name, layout=HDF5.H5D_COMPACT] = conv(val)
      end
    end
  end

  # Create /Data datasets.  All are dimensioned (Npols, Nfreqs, Nblts),
  # with initial dimensions (Npols, NFreqs, 0), i.e. empty,  and max
  # dimensions (Npols, NFreqs, -1), i.e. extensible in the Nblts
  # dimension only.  Chunk size is (Npols, Nfreqs, Nbls).

  # visdata TODO get type from xgpuinfo (or kwarg?)
  create_dataset(data, "visdata", Complex{Int32},
                 ((npols, nfreqs, 0), (npols, nfreqs, -1)),
                 chunk=(npols, nfreqs, nbls),
                 #compress=3 # Only gets 1.5:1 compression
                )

  # flags (use UVH5BoolDatatype to be compatible with Python.Numpy readers)
  create_dataset(data, "flags", UVH5BoolDatatype,
                 ((npols, nfreqs, 0), (npols, nfreqs, -1)),
                 chunk=(npols, nfreqs, nbls), compress=3
                )

  # nsamples
  create_dataset(data, "nsamples", Float32,
                 ((npols, nfreqs, 0), (npols, nfreqs, -1)),
                 chunk=(npols, nfreqs, nbls), compress=3
                )

  # Creation complete!
  uvh5
end

"""
		mkrawname(stem::AbstractString, i::Integer)::String

Make a RAW file name from `stem` and sequency number `i`.  Appends `.IIII.raw`
to `stem`, where `IIII` is `i` zero padded to a width of 4.
"""
function mkrawname(stem::AbstractString, i::Integer)::String
  stem * "." * lpad(i, 4, '0') * ".raw"
end

"""
    findant(name::String,
            antennas::AbstractArray{<:AbstractDict{Symbol,Any},1})::Int

Find index of first element in `antennas` that has `:name` field matching
`name`.  Throws exception if `name` is not found.
"""
function findant(name::String,
                 antennas::AbstractArray{<:AbstractDict{Symbol,Any},1})::Int
  idx = findfirst(a->a[:name]==name, antennas)
  if isnothing(idx)
    error("antenna name $name not found")
  end
  idx
end

"""
    findant(number::Integer,
            antennas::AbstractArray{<:AbstractDict{Symbol,Any},1})::Int

Find index of first element in `antennas` that has `:number` field matching
`number`.  Throws exception if `number` is not found.
"""
function findant(number::Integer,
                 antennas::AbstractArray{<:AbstractDict{Symbol,Any},1})::Int
  idx = findfirst(a->a[:number]==number, antennas)
  if isnothing(idx)
    error("antenna number $number not found")
  end
  idx
end

"""
Use `findant` to map `ant` to index into `antennas` array and convert `pol` to
lowercase Symbols.  Note that this is *not* idempotent.
"""
function mapantpol(ant::Union{String,Integer}, pol::Union{String,Symbol},
                   antennas::AbstractArray{Dict{Symbol,Any},1}
                  )::Tuple{Int,Symbol}
  (findant(ant, antennas), Symbol(lowercase(string(pol))))
end

"""
  extend(ds::HDF5.Dataset, nchunks::Integer=1)
  extend(parent::Union{HDF5.File,HDF5.Group}, ds::String, nchunks::Integer=1)

Extend the size of chunked HDF5.Dataset `ds` or `h5[ds]` by `nchunks` chunks in
the slowest dimension.  Throws exception if `ds` is not chunked or slowest
dimension is not unlimited or a mismatch between an otherwise allowed chunk
size and dataset size.  Specifically, this extends the slowest dimension of
`ds` by `nchunks` times the slowest dimension of chunk size.
"""
function extend(ds::HDF5.Dataset, nchunks::Integer=1)
  chunk = HDF5.get_chunk(ds)
  olddims = size(ds)
  newdims = (olddims[1:end-1]..., olddims[end]+nchunks*chunk[end])
  HDF5.set_extent_dims(ds, newdims)
end

extend(parent::Union{HDF5.File,HDF5.Group}, ds::String, nchunks::Integer=1) = extend(parent[ds], nchunks)

"""
    (in1::Int, in2::Int, ant1::Int, ant2::Int,
     blidx::Int, polidx::Int, isauto::Bool, needsconj::Bool)

An InputPairMapping is a NamedTuple that specifies the mapping between xGPU
input pair `in1` and `in2`, antenna index pair `ant1` and `ant2`, and the
baseline and polarization indexes `blidx` and `polidx` into the `/Data`
datasets of a UVH5 file.  The `isauto` field is true if the input pair is an
autocorrelation.  The `needsconj` field is true if the visibility data needs
conjugation before being written to the UVH5 `visdata` dataset.  See
generateInputPairMap().
"""
const InputPairMapping = @NamedTuple begin
  xgpuidx::Int
  blidx::Int
  polidx::Int
  isauto::Bool
  needsconj::Bool
end

# TODO:
# Reference: https://github.com/david-macmahon/XGPU.jl/blob/main/src/XGPU.jl
# Line 673
function tccInputPairIndex(in1, in2)
  return 7 # Dummy
end

"""
    (pol_array, bl_array, inpair_maps) = generateInputPairMap(antennas, input_map)

For the given `antennas` and `input_map`, create polarization array
`pol_array::Array{Int,1}` containing UVH5/AIPS cross polarization values, a
baseline array `bl_array::Array{Tuple{Int,Int},1}` containing all pairs of all
antenna found in `input_map`, and an input pair mapping list
`inpair_maps::Array{InputPairMapping,1}`.
"""
function generateInputPairMap(antennas::AbstractArray{<:AbstractDict{Symbol,Any},1},
                              input_map::AbstractArray{<:AbstractArray{String,1},1})
  # Make sure we do not have more inputs than supported by xGPU
  num_inputs = length(input_map)
  # @assert num_inputs <= xgpuinfo.nstation * xgpuinfo.npol "input_map too long for xGPU"

  # Remap antennas in input_map from name (String) or number (Int) to index
  # into antennas array and map polarizations to lowercase Symbols
  input_map = map(input_map) do (ant, pol)
    mapantpol(ant, pol, antennas)
  end

  # Determine unique polarizations in input_map.
  pols = sort(unique(getindex.(input_map, 2)))
  npols = length(pols)
  @assert (npols == 1 || npols == 2) "npols must be 1 or 2 (not $npols)"

  # pol_array contains UVH5/AIPS polarization values for all possible
  # polarization pairings.  This is the order in which the polarizations will
  # be stored for a given antenna pair.
  pol_array = [
    POLMAP[p1, p2] for p1 in pols for p2 in pols
  ]

  # pol_map (not to be confused with POLMAP!) is a Dict mapping (:p1,:p2)
  # polarization tuples to "polidx" polarization indexes (i.e. indexes into
  # polarization_array).
  pol_map = Dict(map(i->(POLMAP[pol_array[i]]=>i), eachindex(pol_array)))

  # GUPPI RAW considers each antenna to be dual polarization.  xGPU can
  # be compiled for single or dual polarization inputs.  UVH5 files can
  # be single-polarization or dual-polarization.  In reality, the
  # actual antenna-polarizations going to each input can be a completely
  # arbitrary combination and arrangement of antennas and polarizations.
  #
  # We need to build a list of all possible antenna pairings (aka baselines)
  # given the set of unique antennas that appear in input_map.  It is possible
  # that some baselines will not have all cross polarizations present because
  # some antennas may not have both polarizations present.  The ordering of
  # antennas comprising a baseline is also important.  Because the
  # antenna-polarizations may be connected to arbitrary inputs of the
  # correlator, it is possible that some cross-polarizations will be conjugated
  # differently from other cross-polarizations of the same baseline.  Since
  # there is only one UVW vector for the baseline, these differences must be
  # resolved by conjugating some of the cross-polarizations of these
  # "conflicted" baselines to ensure that all cross-polarizations are
  # conjugated consistently.
  #
  # xGPU computes conj(in1)*in2 where in1 <= in2.  By convention, we constrain
  # baselines (antidx1, antidx2) such that antidx1 <= antidx2.  Note that
  # antidx is the index into the `antennas` array for a given antenna.  This
  # approach relies on the ordering of the antennas array rather than the
  # actual antenna names or numbers, which are essentially opaque labels with
  # no inherent meaning.
  #
  # UVH5 follows the conjugation convention of the Radio Interferometry
  # Measurement Equation (RIME) which defines the visibility product of
  # baseline (antidx1, antidx2) to be ``z_{antidx1}*conj(z_{antidx2})``.
  #
  # If xGPU input pair (in1, in2) with in1 < in2 corresponds to antenna index
  # pair (antidx1, antidx2) with antidx1 <= antidx2, then we consider the
  # visibility product from xGPU to be in need of conjugation before being
  # written to the UVH5 dataset. If we find an xGPU input pair that corresponds
  # to antenna index pair (antidx1, antidx2) with antidx1 > antidx2, then we
  # consider the visibility product be properly conjugated.
  #
  # Auto correlations for dual-polarization systems are handled as a special
  # case, because we write out the extra redundant cross-polarization product.
  # Extra care is required to ensure that these cross-polarization products are
  # properly conjugated.
  #
  # The ordering of the baselines in the dataset is autocorrelations followed
  # by cross correlation.  First we build a baseline array, bl_array, that will
  # be the source of the ant_1_array and ant_2_array contents.  Then we use
  # this list to build a baseline map, `bl_map` that maps (antidx1, antidx2)
  # baselines to a sequential baseline index.  This is used to look up the
  # value for a `bl_idx` field for a given antidx pair (for a given xGPU input
  # pair) when generating the list of InputPairMappings.
  antidxs = sort(unique(getindex.(input_map, 1)))
  nants = length(antidxs)
  nbls = ((nants + 1) * nants) ÷ 2

  bl_array = Array{Tuple{Int,Int},1}(undef, nbls)
  i = 0
  # autos
  for ai in antidxs
    i += 1
    bl_array[i] = (ai, ai)
  end
  # crosses
  for aii1 in 1:nants-1
    for aii2 in aii1+1:nants
      i += 1
      bl_array[i] = (antidxs[aii1], antidxs[aii2])
    end
  end
  # bl_map maps (antidx1, antodx2) tuples to baseline index
  bl_map = Dict(map(i->(bl_array[i]=>i), eachindex(bl_array)))

  # Now we can make list of input pair mappings.
  num_inpairs = (num_inputs+1) * num_inputs ÷ 2
  inpair_maps = Array{InputPairMapping,1}(undef, num_inpairs)
  i = 0
  for in1 in 1:num_inputs
    ant1, pol1 = input_map[in1]
    for in2 in in1:num_inputs
      ant2, pol2 = input_map[in2]
      isauto = (ant1 == ant2)

      # TODO Need to check pol order as well (at least for autos)
      if haskey(bl_map, (ant1, ant2))
        blidx = bl_map[ant1, ant2]
        polidx = pol_map[pol1, pol2]
        needsconj = true
      else
        blidx = bl_map[ant2, ant1]
        polidx = pol_map[pol2, pol1]
        needsconj = false
      end

      i += 1
      inpair_maps[i] = InputPairMapping((tccInputPairIndex(in1, in2),
                                         blidx, polidx, isauto, needsconj))
    end # for in2
  end # for in1

  (pol_array, bl_array, inpair_maps)
end

"""
    calc_uvws!(bl_uvws::AbstractArray{Float64,2},
               antpos_uvw::AbstractArray{Float64,2},
               bl_array::AbstractArray{Tuple{Int,Int},1}
              )::AbstractArray{Float64,2}
"""
function calc_uvws!(bl_uvws::AbstractArray{Float64,2},
                    antpos_uvw::AbstractArray{Float64,2},
                    bl_array::AbstractArray{Tuple{Int,Int},1}
                   )::AbstractArray{Float64,2}
  for (i, (a1, a2)) in enumerate(bl_array)
    # This is the UVH5 convention for baseline UVW direction (i.e. a1->a2)
    bl_uvws[:,i] = antpos_uvw[:,a2] - antpos_uvw[:,a1]
  end

  return bl_uvws
end

mutable struct TCCInfo
  ntime::Int
end

"""
    rawtouvh5(rawstem::AbstractString,
              uvh5name::AbstractString;
              telinfo::Dict{Symbol,Any},
              obsinfo::Dict{Symbol,Any},
              subints_per_dump::Integer=0,
              inttime_per_dump::Real=0
             )::Nothing

Feeds data from GUPPI RAW files corresponding to `rawstem` to xGPU and outputs
visibilities in UVH5 format to file `uvh5name`.  If `rawstem` has a `.NNNN.raw`
suffix, it will be removed.  Metadata for the telescope array is provided in
`telinfo`.  Metadata for the observation that produced the RAW files is
provided in `obsinfo`.  These dictionaries are typically parsed from YAML files
provided by the user.  Integration time is specified by `subints_per_dump`
which is the number of xGPU input buffers to correlate and integrate before
dumping, a value of 0 means calculate from `inttime_per_dump`, values less than
zero mean dump every `-subints_per_dump` blocks.  If both `subints_per_dump`
and `inttime_per_dump` are zero, then dump once per block.  `inttime_per_dump`
must be given in seconds.

A "subint" or "sub-integration" is the integration time granularity set by
XGPU's `xgpuinfo.ntime` value.  The number of time samples per GUPPI RAW block
must be divisible by `xgpuinfo.ntime`.  The integration time of a
sub-integration is given by `TBIN * xgpuinfo.ntime`.

The `telinfo` and `obsinfo` dictionaries are ultimately merged together, so
their separation here is really just for the caller's convenience.  Additional
metadata from the GUPPI RAW header is also added, but not if the caller has
provided explicit values to override the GUPPI RAW header (e.g. corrections to
the source name, sky coordinates, etc.).

Still TODO: Document `telinfo` and `obsinfo` more.
"""
function rawtouvh5(rawstem::AbstractString,
                   uvh5name::AbstractString;
                   telinfo::AbstractDict{Symbol,Any},
                   obsinfo::AbstractDict{Symbol,Any},
                   subints_per_dump::Integer=0,
                   inttime_per_dump::Real=0
                  )::Nothing

  @assert !isfile(uvh5name) "output file $uvh5name exists"
  @assert inttime_per_dump >= 0 "inttime_per_dump cannot be negative"

  # Merge telinfo and obsinfo into metadata
  metadata = merge(telinfo, obsinfo)

  pol_array, bl_array, inpair_maps = generateInputPairMap(metadata[:antennas],
                                                          metadata[:input_map])

  # GUPPI RAW considers each antenna to be dual polarization.  xGPU can
  # be compiled for single or dual polarization inputs.  UVH5 files can
  # be single polarization of dual-polarization.  In reality, the
  # actual antenna-polarizations going to each input can be a completely
  # arbitrary combination and arrangement of antennas and
  # polarizations.  We must determine nants_data from the number of
  # unique antennas that appear in input_map.
  unique_ants = sort(unique(getindex.(metadata[:input_map], 1)))
  nants_data = length(unique_ants)
  nbls = ((nants_data + 1) * nants_data) ÷ 2

  # Set metadata polarization_array
  metadata[:polarization_array] = pol_array
  npols = length(pol_array)

  # Create Arrays for extendable datasets that do NOT depend on RAW headers or
  # data.

  # The a1s and a2s arrays are antenna_numbers, NOT indices (despite the
  # wording in the 2018-11-28 version of the UVH5 memo).  The antenna indexes
  # in bl_array are 1-based indexes into antenna_numbers and antenna_names.
  antenna_numbers = Int.(getindex.(metadata[:antennas], :number))
  a1s = antenna_numbers[getindex.(bl_array, 1)]
  a2s = antenna_numbers[getindex.(bl_array, 2)]
  antpos_enu = getantpos(metadata, :enu)
  # Setup variables for ENU->UVW conversion
  lat_rad = dms2rad(metadata[:latitude])
  lon_rad = dms2rad(metadata[:longitude])
  alt_m = metadata[:altitude]
  antpos_uvw = similar(antpos_enu)
  bl_uvws = Array{Float64,2}(undef, 3, nbls)

  # Trim off .NNNN.raw suffix as a convenience
  rawstem = replace(rawstem, r"\.\d+\.raw$"=>"")
  @info "working stem: $rawstem"

  rawhdr::GuppiRaw.Header = GuppiRaw.Header()
  uvh5 = nothing
  uvh5hdr = nothing
  uvh5data = nothing
  xgpuctx = nothing
  array_h = nothing
  matrix_h = nothing
  rawdata = nothing
  tau = nothing
  taus = nothing
  times = nothing
  lsts = nothing
  flags = nothing
  nsamples = nothing
  visdata = nothing
  nblts = 0
  ntimes = 0
  nfreqs = 0

  ra_rad = 0.0
  dec_rad = 0.0
  recalc_uvws = false

  rawseq = 0
  subints_sent = 0
  subints_per_block = 0

  # TODO: Edit to make realistic
  tccinfo = TCCInfo(16)



  while true
    rawname = mkrawname(rawstem, rawseq)
    if !isfile(rawname)
      @info "opening file: $rawname [not found]"
      break
    end

    

    @info "opening file: $rawname"
    open(rawname) do rawio

      # TODO: Create context for TCC

      # While header
      while read!(rawio, rawhdr)
        if isnothing(uvh5)
          # Validate that RAW file is compatible with xGPU params
          ntime = Blio.ntime(rawhdr)
          # Make sure that RAW ntime is divisible by xGPU NTIME
          # @assert ntime % xgpuinfo.ntime == 0 "RAW ntime not divisible by xgpuinfo.ntime"
          # We currently require that RAW nants <= xGPU NSTATION
          # @assert rawhdr.nants <= xgpuinfo.nstation
          # We currently require that all xGPU inputs are mappable to antpols
          # @assert xgpuinfo.nstation * xgpuinfo.npol >= length(metadata[:input_map])

          subints_per_block = ntime ÷ tccinfo.ntime
          inttime_per_subint = rawhdr.tbin * tccinfo.ntime
          if subints_per_dump == 0
            if inttime_per_subint == 0
              # Default mode of one dump per GUPPI RAW block
              subints_per_dump = subints_per_block
            else
              # Compute subints_per_dump from inttime_per_subint
              subints_per_dump = Int(round(inttime_per_dump/inttime_per_subint))
            end
          elseif subints_per_dump < 0
            subints_per_dump = subints_per_block * -subints_per_dump
          end

          # Calculate actual inttime_per_dump
          inttime_per_dump = inttime_per_subint * subints_per_dump
          @info "integration details" subints_per_block subints_per_dump (
                                      inttime_per_subint) inttime_per_dump

          # Add to metadata based on GUPPI RAW headers unless the caller has
          # provided overrides.

          # Use GUPPI RAW `src_name` header as `object_name` unless the user
          # provided `object_name` in metadata.
          get!(metadata, :object_name, get(rawhdr, :src_name, "UNKNOWN"))
          # TODO Provide more convenient way to allow for freq_array override.
          get!(metadata, :freq_array, collect(chanfreqs(rawhdr)*1e6))
          nfreqs = length(metadata[:freq_array])
          # We currently require that nfreqs <= xGPU nfrequency
          # @assert nfreqs <= xgpuinfo.nfrequency "RAW nfreqs ($nfreqs) > XGPU nfrequency ($(xgpuinfo.nfrequency))"

          get!(metadata, :channel_width, abs(get(rawhdr, :chan_bw, 0)) * 1e6)
          if !isa(metadata[:channel_width], AbstractVector)
            metadata[:channel_width] = fill(metadata[:channel_width], nfreqs)
          end

          # If RAW file has RA and DEC values in the header, phase_type is
          # "phased", otherwise it is "drift".
          if haskey(rawhdr, :ra) && haskey(rawhdr, :dec)
            metadata[:phase_type] = "phased"
            ra_rad = get!(metadata, :phase_center_ra, deg2rad(rawhdr.ra))
            dec_rad = get!(metadata, :phase_center_dec, deg2rad(rawhdr.dec))
            # TODO Handle case when phase_center_epoch is not 2000.0
            get!(metadata, :phase_center_epoch, 2000.0)
            get!(metadata, :phase_center_frame, "icrs")
            recalc_uvws = true
          else
            # Drift scan is phased to zenith, so antpos_uvw is identical to antpos_enu
            metadata[:phase_type] = "drift"
            calc_uvws!(bl_uvws, antpos_enu, bl_array)
            recalc_uvws = false
          end

          # Create and initialize uvh5 file
          uvh5 = uvh5create(uvh5name; metadata...)
          uvh5hdr = uvh5["Header"]
          uvh5data = uvh5["Data"]

          # Init xGPU and get context
          xgpuctx=xgpuInit()
          array_h=xgpuInputArray(xgpuctx);
          matrix_h=xgpuOutputArray(xgpuctx);

          # Create rawdata Array
          rawdata = Array(rawhdr)

          # Create Arrays for extendable datasets that DO depend on RAW headers
          tau = Float64(inttime_per_dump)
          taus = fill(tau, nbls)
          # Initialize times to half of tau *before* start time of file because
          # each dump will increment times by tau.
          # TODO Use RAW SYNCTIME,PIPERBLK,TBIN,ntime (if all available),
          # otherwise use STT_IMJD,STT_SMJD, otherwise use DAQPULSE, otherwise
          # use ???
          t0 = MJD0 + get(rawhdr, :stt_imjd, 0) +
               (get(rawhdr, :stt_smjd, 0) - tau/2) / ERFA.DAYSEC
          times = fill(t0, nbls)
          lsts = Array{Float64}(undef, nbls)
          flags = zeros(Int8, npols, nfreqs, nbls)
          nsamples = ones(Float64, npols, nfreqs, nbls)
          visdata = zeros(eltype(matrix_h), npols, nfreqs, nbls)
        end # if isnothing(uvh5) # lazy init

        # Read block
        read!(rawio, rawdata)

        # TODO: Run TCC

#=
        # Debug, zero out data, then set test values
        rawdata .= 0
        rawdata[1,1,1,1] = 1+2im # indexed as (pol, time, chan, ant)
        rawdata[2,1,1,1] = 3+4im # indexed as (pol, time, chan, ant)
        rawdata[2,1,1,2] = 3+4im # indexed as (pol, time, chan, ant)
=#

        # For subint in block
        for i in 1:subints_per_block
          # Swizzle subint data into XGPU input buffer array_h
          #rawSwizzleInput!(array_h, rawdata, xgpuinfo.ntime*(i-1)+1, xgpuinfo,
                          #  rawhdr.nants, nfreqs)
          # send subint to GPU
          #xgpuCudaXengine(xgpuctx, XGPU.SYNCOP_SYNC_TRANSFER)
          # Count subints
          subints_sent += 1

          # If nsent == subints_per_dump
          if subints_sent % subints_per_dump == 0
            # Dump integration buffer to matrix_h
            xgpuDumpDeviceIntegrationBuffer(xgpuctx)
            # Clear integration buffer on device
            xgpuClearDeviceIntegrationBuffer(xgpuctx)

            # Increment times
            times .+= tau / ERFA.DAYSEC

            # Get interpolated ut1utc value
            ut1utcsec = getΔUT1(times[1])
            ut1utcday = ut1utcsec / ERFA.DAYSEC

            # Fill lsts with calculated LST.  This uses UT1 for both UT1 and
            # TT, which results in an error on the order of 100 microarcseconds
            # or approximately 7 microseconds.
            lsts .= ERFA.gst06a(times[1], ut1utcday, times[1], ut1utcday) + lon_rad

            # Update uvws if we have the required info
            if recalc_uvws
              aob, zob, hob, dob, rob, eo = ERFA.atco13(
                 ra_rad, dec_rad,  # right ascension, declination
                 0, 0, 0, 0,       # proper motion ra/dec, parallax, rv,
                 times[1], 0,      # jd1 (UTC), jd2 (UTC)
                 ut1utcsec,        # DUT1 (UT1-UTC) at times[1]
                 lon_rad, lat_rad, alt_m, # observer location
                 0, 0,             # polar motion (UVH5 does not record this)
                 0, 0, 0, 0)       # atmosphereic refraction

              enu2uvw!(antpos_uvw, antpos_enu, hob, dob, lat_rad)

              calc_uvws!(bl_uvws, antpos_uvw, bl_array)
            end

            # Extend datasets
            extend(uvh5hdr["ant_1_array"])
            extend(uvh5hdr["ant_2_array"])
            extend(uvh5hdr["uvw_array"])
            extend(uvh5hdr["time_array"])
            extend(uvh5hdr["lst_array"])
            extend(uvh5hdr["integration_time"])
            extend(uvh5data["flags"])
            extend(uvh5data["nsamples"])
            extend(uvh5data["visdata"])

            # Update non-scalar Header datasets
            uvh5hdr["ant_1_array"][(1:nbls).+nblts] = a1s
            uvh5hdr["ant_2_array"][(1:nbls).+nblts] = a2s
            uvh5hdr["uvw_array"][:,(1:nbls).+nblts] = bl_uvws
            uvh5hdr["time_array"][(1:nbls).+nblts] = times
            uvh5hdr["lst_array"][(1:nbls).+nblts] = lsts
            uvh5hdr["integration_time"][(1:nbls).+nblts] = taus

            # Update Data datasets
            uvh5data["flags"][:, :, (1:nbls).+nblts] = flags
            uvh5data["nsamples"][:, :, (1:nbls).+nblts] = nsamples
            for (xgpuidx, blidx, polidx, isauto, needsconj) in inpair_maps
                visdata[polidx, :, blidx] = (
                    needsconj ? conj.(matrix_h[xgpuidx, 1:nfreqs])
                              :       matrix_h[xgpuidx, 1:nfreqs])
                # If cross-pol autocorrelation
                if isauto && polidx == 2
                  # Output redundant cross-pol autocorrelation
                  # We could cheat and output conjugate of current xgpuidx
                  # spectrum, but instead we use some inside knowledge that
                  # should be publicized in XGPU documentation that the
                  # redundant cross-pol is at xgpuidx-1.

                  # TODO: Figure out what to do here

                  # visdata[polidx+1, :, blidx] = (
                  #     needsconj ? conj.(matrix_h[xgpuidx-1, 1:nfreqs])
                  #               :       matrix_h[xgpuidx-1, 1:nfreqs])
                end
            end
            uvh5data["visdata"][:, :, (1:nbls).+nblts] = visdata

            # Update scalar datasets in Header
            nblts += nbls
            ntimes += 1
            write(uvh5hdr["Nblts"], nblts)
            write(uvh5hdr["Ntimes"], ntimes)

#=
            # Debug, exit after first dump
            close(uvh5)
            xgpuFree(xgpuctx)
            exit(1)
=#

          end # if dump time
        end # for subint in block
      end # while read header
    end # open rawname

    rawseq += 1
  end # "forever" loop

  @info "output $ntimes integrations"
  @info "$(subints_sent%subints_per_dump)/$subints_per_dump leftover"

  # Close uvh5
  if uvh5 !== nothing
    close(uvh5)
  end

  # TODO: Free TCC

  # Free XGPU resources
  # if xgpuctx !== nothing
  #   xgpuFree(xgpuctx)
  # end

end # rawtouvh5()

end # module
