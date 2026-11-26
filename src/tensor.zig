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
        buffer_t,
        // favor single buffer of data
        buffer_u8,
        // TODO buffer_bit ?
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

    backend: Backend = .cpu,

    pub fn dimensions(def: TensorDef) comptime_int {
        return def.shape.len;
    }

    pub fn Data(def: TensorDef) type {
        return switch (def.backend) {
            .cpu => switch (def.implementation) {
                .arrays => Arrays(def),
                .buffer_t => Buffer(def),
                .buffer_u8 => @compileLog("unimplemented"),
            },
            .spirv => @compileError("unimplemented"),
            .rocm => @compileError("unimplemented"),
            .cuda => @compileError("unimplemented"),
        };
    }
};

fn Arrays(def: TensorDef) type {
    // TODO a comptime assert rather
    if (def.implementation != .arrays) @compileError("unreachable");
    return struct {
        const Self = @This();
        pub const Data = ArraysRec(def.shape, def.Type);
        data: Data,
        pub fn all(value: def.Type) Self {
            comptime {
                if (def.shape.len == 0) return .{ .data = .{} };
            }
            const all_rec = struct {
                fn allRec(comptime shape: []const comptime_int, v: def.Type, a: *ArraysRec(shape, def.Type)) void {
                    if (comptime shape.len == 1) {
                        a.* = @splat(v);
                        return;
                    }
                    inline for (0..shape[0]) |i|
                        allRec(shape[1..], v, &a[i]);
                    return;
                }
            };
            var result: Self = undefined;
            all_rec.allRec(def.shape, value, &result.data);
            return result;
        }
    };
}

fn ArraysRec(shape: []const comptime_int, T: type) type {
    if (shape.len == 0) return void;
    if (shape.len == 1) return [shape[0]]T;
    return [shape[0]]ArraysRec(shape[1..], T);
}

fn Buffer(def: TensorDef) type {
    // TODO a comptime assert rather
    if (def.implementation != .buffer_t) @compileError("unreachable");
    return struct {
        const Self = @This();
        pub const Data = BufferRec(def.shape, def.Type);
        data: Data,

        pub fn all(value: def.Type) Self {
            return .{ .data = @splat(value) };
        }
    };
}

fn BufferRec(shape: []const comptime_int, T: type) type {
    if (shape.len == 0) return void;
    var dim = 1;
    for (shape) |s| dim *= s;
    return [dim]T;
}

fn Slice(dimension: comptime_int, T: type) type {
    if (dimension == 0) return void;
    if (dimension == 1) return []const T;
    return []Slice(dimension - 1, T);
}

pub fn Tensor(_def: TensorDef) type {
    for (_def.shape) |s| if (s == 0) @compileError("Tensor with a zero length dimension not allowed");
    return struct {
        const Self = @This();
        pub const def = _def;
        pub const Data = def.Data();
        pub const T = def.Type;

        data: Data,

        pub fn all(value: T) Self {
            return .{ .data = .all(value) };
        }

        pub fn fromSlice(values: Slice(def.dimensions(), def.Type)) Self {
            _ = values; // autofix
            // TODO code from pytorch?? -> mem copy likely need to work with ssd init
            const v: Self = undefined;
            // TODO also maybe put impl under its own struct impl
            return v;
        }
    };
}

pub const tests = struct {
    const std = @import("std");
    const expect = std.testing.expect;
    const expectEqual = std.testing.expectEqual;

    test "instantiate tensor" {
        const S0 = Tensor(.{ .shape = &.{} });
        try expectEqual(0, S0.def.dimensions());
        try expectEqual(void, S0.Data.Data);
        const S1 = Tensor(.{ .shape = &.{5} });
        try expectEqual(1, S1.def.dimensions());
        try expectEqual([5]u64, S1.Data.Data);
        const S2 = Tensor(.{ .shape = &.{ 4, 3 } });
        try expectEqual(2, S2.def.dimensions());
        try expectEqual([4][3]u64, S2.Data.Data);
        const S3 = Tensor(.{ .shape = &.{ 4, 2, 5 } });
        try expectEqual(3, S3.def.dimensions());
        try expectEqual([4][2][5]u64, S3.Data.Data);
        const S3b = Tensor(.{ .shape = &.{ 4, 2, 5 }, .implementation = .buffer_t });
        try expectEqual(3, S3b.def.dimensions());
        try expectEqual([4 * 2 * 5]u64, S3b.Data.Data);

        try expectEqual([][][]const u8, Slice(3, u8));

        const s3b = S3b.all(3);
        try expectEqual(3, s3b.data.data[0]);
        try expectEqual(3, comptime S3b.all(3).data.data[0]);

        const s3 = S3.all(3);
        try expectEqual(3, s3.data.data[0][0][0]);
        try expectEqual(3, comptime S3.all(3).data.data[0][0][0]);

        // TODO try comptime instantiation fromSlice
    }
};
