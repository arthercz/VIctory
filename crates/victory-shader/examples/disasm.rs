//! Disassemble an Xbox shader container's native instructions.
//! cargo run --example disasm -- <file.dxbc> [offset]

use victory_shader::{Container, FourCC, native_programs};

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: disasm <file.dxbc>");
        std::process::exit(2);
    };
    let bytes = std::fs::read(&path).expect("read");
    let container = Container::parse(&bytes).expect("container");
    let dxil = container.part(FourCC::DXIL).expect("dxil part");
    for (i, program) in native_programs(&dxil.data)
        .expect("programs")
        .iter()
        .enumerate()
    {
        println!(
            "== program {i}: {:?} wave64={}",
            program.stage, program.wave64
        );
        let code = program.patched_code();
        let insts = victory_shader::gfx10::decode(&code).expect("decode");
        for inst in &insts {
            println!("{}", inst.text(program.wave64));
        }
    }
}
