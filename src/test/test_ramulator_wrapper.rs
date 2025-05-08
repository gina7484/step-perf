
#[cfg(test)]
mod test {
    use ramulator_wrapper::RamulatorWrapper;

    #[test]
    fn instantiate_wrapper(){
        println!("Instantiateing the wrapper");
        let ramulator =
            RamulatorWrapper::new("/home/ginasohn/step-perf/external/ramulator2_wrapper/configs/hbm3.yaml");
        println!("{}",ramulator.get_cycle());
        println!("Ramulator wrapper instantiated");
    }
}