#[cfg(test)]
mod test {
    use symbolica::{atom::Atom, parse, symbol};

    #[test]
    fn deserialize_symbol() {
        // Method 1: Create symbols programmatically
        let x = symbol!("x");
        let y = symbol!("y");
        let z = symbol!("z"); // New symbol

        // Convert symbols to atoms
        let x_atom: Atom = x.into();
        let y_atom: Atom = y.into();
        let z_atom: Atom = z.into();

        // Build expressions programmatically
        let expr1 = &x_atom * &x_atom; // x^2
        let expr2 = &y_atom * &y_atom; // y^2
        let combined = &expr1 + &expr2; // x^2 + y^2

        // Add the new symbol z
        let final_expr = &combined + &z_atom; // x^2 + y^2 + z

        println!("Final expression: {}", final_expr);

        // Method 2: Parse existing expression and add new terms
        // let parsed_expr = parse!("x^2 + 2*x*y + y^2").unwrap();
        // let parsed_expr: Atom = parse!("floor(s0/16 + 15/16)").unwrap();
        let string_expr = "floor(s0/16 + 15/16)".to_string();
        let parsed_expr: Atom = parse!(&string_expr).unwrap();
        let new_term: Atom = &z_atom * &z_atom; // z^2
        let modified_expr: Atom = &parsed_expr + &new_term;

        println!("Modified expression: {}", modified_expr);
    }
}
