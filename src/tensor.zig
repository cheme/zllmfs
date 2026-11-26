//! tensor definition
//! goals:
//! - independant of backend: define as n dim zig array
//!     - cpu comptime
//!     - cpu on stack
//!     - cpu allocated
//!     - gpu spirv compute
//!     - gpu rocm
//!     - gpu cuda
//! - streamed instantiation to avoid passing cpu memory on gpu backend, or just save memory.
//! - remote backend: segmented large tensor access (allow running on small memory without os virtualizing large mem).
//!
//! As usual very partial implementation based on use.

pub const TensorDef = struct {
    pub const Implementation = enum {
        // favor multiple array
        // for cpu zig arrays
        arrays,
        // favor single buffer of data
        buffer,
    };

    pub const Backend = enum {
        cpu,
        spirv,
        rocm,
        cuda,
    };

    shape: []const comptime_int,
    // TODO default to default arch int
    // or TODO no default want things explicit.
    Type: type = u64,

    implementation: Implementation = .arrays,

    prefer_packed: bool = false,

    backend: Backend = .cpu,

    pub fn dimensions(def: TensorDef) comptime_int {
        return def.shape.len;
    }

    pub fn Data(def: TensorDef) type {
        return switch (def.backend) {
            .cpu => switch (def.implementation) {
                .arrays => Arrays(def.shape, def.Type),
                .buffer => Buffer(def.shape, def.Type),
            },
            .spirv => @compileError("unimplemented"),
            .rocm => @compileError("unimplemented"),
            .cuda => @compileError("unimplemented"),
        };
    }
};

fn Arrays(shape: []const comptime_int, T: type) type {
    if (shape.len == 0) return void;
    if (shape.len == 1) return [shape[0]]T;
    return [shape[0]]Arrays(shape[1..], T);
}

fn Buffer(shape: []const comptime_int, T: type) type {
    if (shape.len == 0) return void;
    var dim = 1;
    for (shape) |s| dim *= s;
    return [dim]T;
}

pub fn Tensor(_def: TensorDef) type {
    for (_def.shape) |s| if (s == 0) @compileError("Tensor with a zero length dimension not allowed");
    return struct {
        pub const def = _def;
        pub const Data = def.Data();

        data: def.Data(),
    };
}

pub const tests = struct {
    const std = @import("std");
    const expect = std.testing.expect;
    const expectEqual = std.testing.expectEqual;

    test "instantiate tensor" {
        const S0 = Tensor(.{ .shape = &.{} });
        try expectEqual(0, S0.def.dimensions());
        try expectEqual(void, S0.Data);
        const S1 = Tensor(.{ .shape = &.{5} });
        try expectEqual(1, S1.def.dimensions());
        try expectEqual([5]u64, S1.Data);
        const S2 = Tensor(.{ .shape = &.{ 4, 3 } });
        try expectEqual(2, S2.def.dimensions());
        try expectEqual([4][3]u64, S2.Data);
        const S3 = Tensor(.{ .shape = &.{ 4, 2, 5 } });
        try expectEqual(3, S3.def.dimensions());
        try expectEqual([4][2][5]u64, S3.Data);
        const S3b = Tensor(.{ .shape = &.{ 4, 2, 5 }, .implementation = .buffer });
        try expectEqual(3, S3b.def.dimensions());
        try expectEqual([4 * 2 * 5]u64, S3b.Data);
    }
};
