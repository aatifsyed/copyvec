<!-- cargo-rdme start -->

A stack-allocated sequence that mirror's [`Vec`](https://doc.rust-lang.org/std/vec/struct.Vec.html)'s API,
but:
- Implements [`Copy`] (and can only hold [`Copy`] types).
- Does not grow.
- Is `#[no_std]`/no-`alloc` compatible.

```rust

// const-friendly
const VEC: CopyVec<&str, 10> = CopyVec::new();

// easy initialising
let mut vec = copyvec!["a", "b", "c"; + 2];
                                   // ^~ with excess capacity

// use the API's you know
vec.push("d");

// including iteration
for it in &mut vec {
    if *it == "a" {
        *it = "A"
    }
}

assert_eq!(vec, ["A", "b", "c", "d"]);
vec.retain(|it| *it == "b" || *it == "c")
assert_eq!(vec.remove(0), "b");
```

# Other features
- [`serde`](https://docs.rs/serde/)
- [`quickcheck`](https://docs.rs/quickcheck/)


If you like this crate, you may also enjoy [`stackstack`](https://docs.rs/stackstack)

<!-- cargo-rdme end -->
