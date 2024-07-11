<!-- cargo-rdme start -->

A stack-allocated sequence that mirror's [`Vec`](https://doc.rust-lang.org/std/vec/struct.Vec.html)'s API,
but:
- Implements [`Copy`] (and can only hold [`Copy`] types).
- Does not grow.
- Is `#[no_std]`/no-`alloc` compatible.

<!-- cargo-rdme end -->
