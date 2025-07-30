use crabby_monkeys::simulators::simulate_trade_partial;

fn main() {
    let prices = vec![1.0; 400];      // dummy 20*20 bids
    let (px, open, idx) = simulate_trade_partial(&prices, 1.0, 10);
    println!("{px} {open} {idx}");
}