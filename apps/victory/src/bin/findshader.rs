//! One-off: find a shader container by blake3 prefix (16 hex) in files,
//! including SGD5-packed records, and write the container bytes out.
//! Usage: victory-findshader <prefix> <out> <files...>

use std::borrow::Cow;

fn main() {
    let mut args = std::env::args().skip(1);
    let prefix = args.next().expect("hash prefix").to_ascii_lowercase();
    let out = args.next().expect("out path");
    let files: Vec<_> = args.collect();
    assert!(!files.is_empty(), "no files");
    for file in &files {
        let data = std::fs::read(file).expect("read");
        eprintln!("{file}: {} bytes", data.len());
        let mut found = 0usize;
        let mut check = |bytes: &[u8], where_: &str| {
            let Ok(container) = victory_shader::Container::parse(bytes) else {
                return;
            };
            found += 1;
            let h = blake3::hash(bytes).to_hex();
            if h.starts_with(&prefix) {
                eprintln!("MATCH {where_}: hash {h}, {} bytes", bytes.len());
                std::fs::write(&out, bytes).expect("write");
                eprintln!("written to {out}");
                if let Ok(c) = victory_shader::Container::parse(bytes)
                    && let Some(p) = c.part(victory_shader::FourCC::DXIL)
                    && let Ok(programs) = victory_shader::native_programs(p.data)
                {
                    for n in &programs {
                        eprintln!(
                            "  stage {:?} wave64 {} code {} B io {:?}",
                            n.stage,
                            n.wave64,
                            n.code.len(),
                            n.io
                        );
                    }
                }
                std::process::exit(0);
            }
            let _ = container;
        };
        if victory_game::sgd5::is_sgd5(&data) {
            for r in victory_game::sgd5::records(&data) {
                let name = r.name.unwrap_or("?").to_owned();
                let off = r.offset;
                if let Some(bytes) = r.unpack() {
                    let owned: Cow<[u8]> = Cow::Owned(bytes);
                    check(&owned, &format!("{file} sgd5 @{off:#x} {name}"));
                }
            }
        }
        for i in memchr::memmem::find_iter(&data, b"DXBC") {
            check(&data[i..], &format!("{file} raw @{i:#x}"));
        }
        eprintln!("{file}: {found} containers, no match");
    }
    eprintln!("not found");
    std::process::exit(1);
}
