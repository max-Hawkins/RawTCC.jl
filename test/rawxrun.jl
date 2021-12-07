#!/bin/bash
#=
exec julia --color=yes --startup-file=no --project=`dirname "${BASH_SOURCE[0]}"`/.. "${BASH_SOURCE[0]}" "$@"
=#

import YAML

using Rawx

cd(@__DIR__)
telinfo = YAML.load_file("telinfo.yml", dicttype=Dict{Symbol,Any})
obsinfo = YAML.load_file("obsinfo.yml", dicttype=Dict{Symbol,Any})
subints_per_dump = 8

uvh5file = "blk2.uvh5"
if isfile(uvh5file)
  @info "removing existing output file $uvh5file"
  rm(uvh5file)
end

# Uses Julia 1.5 bare keyword argument feature!
Rawx.rawtouvh5("blk2", "blk2.uvh5"; telinfo, obsinfo, subints_per_dump)
