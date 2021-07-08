use crate::*;
use proc_macro2::TokenStream;
use syn::*;
use syn::parse::{Parse, ParseStream, Parser};
use quote::{ToTokens, quote};


#[derive(Default, Copy, Clone)]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct GlobalSettings {
  deterministic: bool,
  skip_regressions: bool,
  debug: bool,
}

impl UpdateFromMeta for GlobalSettings {
  fn update(&mut self, meta: &Meta) -> Result<bool> {
    match meta {
      Meta::Path(p) => {
        if p.is_ident("skip_regressions") {
          self.skip_regressions = true;
        } else if p.is_ident("debug") {
          self.debug = true;
        } else if p.is_ident("deterministic") {
          self.deterministic = true;
        } else {
          return Ok(false);
        }
        Ok(true)
      }
      _ => Ok(false)
    }
  }
}

impl_parse_for_updatefrommeta!(GlobalSettings);


#[derive(Default, Clone)]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct Config {
  viz: VizConfig,
  cpus: Option<LitInt>,
  cases: Option<LitInt>,
}

fn expect_litint(lit: &Lit) -> Result<&LitInt> {
  match lit {
    Lit::Int(l) => Ok(l),
    _ => parse_error!(lit => "expected integer literal"),
  }
}


impl UpdateFromMeta for Config {
  fn update(&mut self, meta: &Meta) -> Result<bool> {
    if self.viz.update(meta)? {
      return Ok(true);
    }
    match meta {
      Meta::NameValue(kwarg) => {
        if kwarg.path.is_ident("cpus") {
          self.cpus = Some(expect_litint(&kwarg.lit)?.clone())
        } else if kwarg.path.is_ident("cases") {
          self.cases = Some(expect_litint(&kwarg.lit)?.clone())
        } else {
          return Ok(false);
        }
        Ok(true)
      }
      _ => Ok(false)
    }
  }
}

impl_parse_for_updatefrommeta!(Config);

impl ToTokens for Config {
  fn to_tokens(&self, tokens: &mut TokenStream) {
    let mut config_block = self.viz.to_token_stream();
    if let Some(n) = &self.cpus {
      config_block.extend(quote! {
        runner.cpus(#n);
      })
    }
    if let Some(n) = &self.cases {
      config_block.extend(quote! {
        runner.cases(#n);
      })
    }
    tokens.extend(quote! {{ #config_block }});
  }
}

#[derive(Clone)]
#[cfg_attr(feature = "debug", derive(Debug))]
pub enum ProptestDirective {
  Config(Config),
  Input(TokenStream),
}

fn strip_parens(input: ParseStream) -> Result<TokenStream> {
  let content;
  parenthesized!(content in input);
  content.parse()
}

impl Directive for ProptestDirective {
  fn from_attribute(attr: &Attribute) -> Result<Option<Self>> {
    let val = if attr.path.is_ident("config") {
      Some(ProptestDirective::Config(attr.parse_args()?))
    } else if attr.path.is_ident("input") {
      let s = strip_parens.parse2(attr.tokens.clone())?;
      Some(ProptestDirective::Input(s))
    } else {
      None
    };
    Ok(val)
  }
}

impl ToTokens for ProptestDirective {
  fn to_tokens(&self, tokens: &mut TokenStream) {
    use ProptestDirective::*;
    match self {
      Input(ts) => {
        tokens.extend(quote! {
          runner.run(#ts);
        })
      }
      Config(config) =>
        config.to_tokens(tokens),
    }
  }
}

pub type GraphProptest = Test<ProptestDirective, GlobalSettings>;

impl Parse for GraphProptest {
  fn parse(input: ParseStream) -> Result<Self> {
    Test::parse(input, "GraphProptestResult")
  }
}

impl ToTokens for GraphProptest {
  fn to_tokens(&self, tokens: &mut TokenStream) {
    let GraphProptest {
      global_settings,
      outer_attrs,
      directives,
      fn_ident,
      fn_sig,
      fn_wrapper,
      fn_body
    } = self;
    let global_settings = global_settings.unwrap_or_default();

    let directives = if global_settings.debug {
      quote! { runner.debug(); }
    } else {
      let deterministic = if global_settings.deterministic {
        Some(quote! { runner.deterministic(true ); })
      } else {
        None
      };
      let skip_regressions = if global_settings.skip_regressions {
        Some(quote! { runner.skip_regressions(true); })
      } else {
        None
      };
      quote! {
        #skip_regressions
        #deterministic
        #(#directives)*
      }
    };

    let inner_ident = &fn_sig.ident;

    let t = quote! {
      #[test]
      fn #fn_ident() {
        #(#outer_attrs)*
        #fn_sig
        #fn_body

        let mut runner = crate::test::GraphProptestRunner::new(
          crate::test::get_test_id(module_path!(), stringify!(#fn_ident)),
          #fn_wrapper(#inner_ident)
        );

        #directives
      }
    };
    tokens.extend(t);
  }
}
